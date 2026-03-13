from __future__ import annotations

import json
import shutil
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .audio import assemble_wav_files, extract_reference_excerpts, write_wav_file
from .backend import VoiceBackend
from .quality_check import SegmentQualityReport, analyze_segment_quality
from .settings import (
    DEFAULT_REFERENCE_EXCERPT_COUNT,
    DEFAULT_REFERENCE_EXCERPT_SECONDS,
    DEFAULT_OUTPUT_SAMPLE_RATE,
    DEFAULT_SYNTHESIS_SEED,
    JOBS_DIR,
    MAX_REFERENCE_SECONDS,
    MAX_RESYNTHESIS_ATTEMPTS,
    ensure_app_directories,
)
from .text_processing import build_synthesis_chunks
from .validation import ValidationError, validate_reference_wav, validate_spanish_text


@dataclass(frozen=True)
class SynthesisResult:
    final_audio_path: str
    segment_count: int
    reference_paths_used: list[str]
    operation_note: str | None
    job_dir: str
    quality_issues: list[str] = field(default_factory=list)
    resynthesized_count: int = 0


class SpanishVoiceService:
    def __init__(
        self,
        backend: VoiceBackend,
        jobs_dir: Path | None = None,
    ) -> None:
        self.backend = backend
        self.jobs_dir = jobs_dir or JOBS_DIR
        self._last_operation_note: str | None = None
        ensure_app_directories()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_spanish(
        self,
        reference_audio_path: str | None,
        text_es: str,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> SynthesisResult:
        normalized_text = validate_spanish_text(text_es)
        chunks = build_synthesis_chunks(normalized_text)

        if not chunks:
            raise ValidationError("No se pudo obtener ningun fragmento de texto valido para la sintesis.")

        total_steps = len(chunks) + 2
        self._notify_progress(progress_callback, "Preparando referencia...", 0, total_steps)
        job_dir = self._build_job_dir()
        segments_dir = job_dir / "segments"
        references_dir = job_dir / "references"
        segments_dir.mkdir(parents=True, exist_ok=True)
        references_dir.mkdir(parents=True, exist_ok=True)

        reference_paths = self._prepare_reference_audio(reference_audio_path, references_dir)
        manifest_path = job_dir / "manifest.json"
        final_audio_path = job_dir / "final.wav"
        manifest = {
            "job_id": job_dir.name,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "job_dir": str(job_dir),
            "final_audio_path": str(final_audio_path),
            "reference_paths_used": reference_paths,
            "operation_note": self._last_operation_note,
            "segment_count": len(chunks),
            "segments": [
                {
                    "index": index,
                    "path": str(segments_dir / f"segment_{index:03d}.wav"),
                    "char_count": len(chunk),
                    "text_preview": chunk[:120],
                    "status": "pending",
                }
                for index, chunk in enumerate(chunks, start=1)
            ],
        }
        self._write_manifest(manifest_path, manifest)

        segment_paths: list[str] = []
        quality_issues: list[str] = []
        resynthesized_count = 0

        try:
            for index, chunk in enumerate(chunks, start=1):
                self._notify_progress(progress_callback, f"Sintetizando bloque {index}/{len(chunks)}...", index, total_steps)

                segment_path = segments_dir / f"segment_{index:03d}.wav"
                success, attempts, issues = self._synthesize_with_quality_check(
                    text=chunk,
                    reference_paths=reference_paths,
                    segment_path=segment_path,
                    base_seed=DEFAULT_SYNTHESIS_SEED + index,
                )

                if attempts > 1:
                    resynthesized_count += 1
                    quality_issues.append(f"Bloque {index}: re-sintetizado ({attempts} intentos)")

                if issues:
                    for issue in issues:
                        quality_issues.append(f"Bloque {index}: {issue}")

                segment_paths.append(str(segment_path))
                manifest["segments"][index - 1]["status"] = "completed"
                manifest["segments"][index - 1]["attempts"] = attempts
                self._write_manifest(manifest_path, manifest)

            self._notify_progress(progress_callback, "Ensamblando audio final...", len(chunks) + 1, total_steps)
            assemble_wav_files(segment_paths=segment_paths, target_path=final_audio_path, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE)
            manifest["status"] = "completed"
            manifest["completed_at"] = datetime.now().isoformat()
            manifest["quality_issues"] = quality_issues
            manifest["resynthesized_count"] = resynthesized_count
            self._write_manifest(manifest_path, manifest)
        except Exception as exc:
            failed_index = len(segment_paths) + 1
            manifest["status"] = "failed"
            manifest["failed_segment"] = failed_index
            manifest["error"] = str(exc)
            if 0 <= failed_index - 1 < len(manifest["segments"]):
                manifest["segments"][failed_index - 1]["status"] = "failed"
            self._write_manifest(manifest_path, manifest)
            raise RuntimeError(f"Fallo en bloque {failed_index}/{len(chunks)}: {exc}") from exc

        self._notify_progress(progress_callback, "Listo.", total_steps, total_steps)
        return SynthesisResult(
            final_audio_path=str(final_audio_path),
            segment_count=len(segment_paths),
            reference_paths_used=reference_paths,
            operation_note=self._last_operation_note,
            job_dir=str(job_dir),
            quality_issues=quality_issues,
            resynthesized_count=resynthesized_count,
        )

    def _synthesize_with_quality_check(
        self,
        text: str,
        reference_paths: list[str],
        segment_path: Path,
        base_seed: int,
    ) -> tuple[bool, int, list[str]]:
        """
        Sintetiza un segmento con verificación de calidad y re-síntesis automática.

        Returns:
            Tupla (éxito, número_de_intentos, problemas_detectados).
        """
        all_issues: list[str] = []

        for attempt in range(1, MAX_RESYNTHESIS_ATTEMPTS + 1):
            seed = base_seed + (attempt - 1) * 1000
            rendered_audio = self.backend.synthesize_fragment(
                text=text,
                reference_audio_paths=reference_paths,
                seed=seed,
            )
            write_wav_file(segment_path, rendered_audio, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE)

            report = analyze_segment_quality(segment_path, text)

            if report.is_valid:
                return True, attempt, all_issues

            all_issues.extend(report.issues)

            if attempt < MAX_RESYNTHESIS_ATTEMPTS:
                continue

        return False, MAX_RESYNTHESIS_ATTEMPTS, all_issues

    def describe_runtime(self) -> str:
        return self.backend.describe_runtime()

    def get_last_operation_note(self) -> str | None:
        return self._last_operation_note

    def _build_job_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.jobs_dir / f"job_{timestamp}_{uuid4().hex[:8]}"

    def _prepare_reference_audio(self, reference_audio_path: str | None, references_dir: Path) -> list[str]:
        duration_seconds = validate_reference_wav(reference_audio_path)
        self._last_operation_note = None

        if duration_seconds <= MAX_REFERENCE_SECONDS:
            source_path = Path(reference_audio_path)
            target_path = references_dir / f"reference_01{source_path.suffix.lower() or '.wav'}"
            shutil.copy2(source_path, target_path)
            return [str(target_path)]

        excerpts = extract_reference_excerpts(
            source_path=reference_audio_path,
            target_dir=references_dir,
            excerpt_seconds=DEFAULT_REFERENCE_EXCERPT_SECONDS,
            excerpt_count=DEFAULT_REFERENCE_EXCERPT_COUNT,
        )
        if not excerpts:
            raise ValidationError("No se pudieron preparar referencias validas a partir del audio subido.")

        start_seconds = ", ".join(f"{excerpt.start_seconds:.1f}" for excerpt in excerpts)
        self._last_operation_note = (
            f"El audio de referencia duraba {excerpts[0].original_duration_seconds:.1f} s. "
            f"Se han usado automaticamente {len(excerpts)} tramos de "
            f"{excerpts[0].excerpt_duration_seconds:.1f} s en los segundos {start_seconds}."
        )
        return [str(excerpt.path) for excerpt in excerpts]

    def _notify_progress(
        self,
        progress_callback: Callable[[str, int, int], None] | None,
        status: str,
        step: int,
        total_steps: int,
    ) -> None:
        if progress_callback is not None:
            progress_callback(status, step, total_steps)

    def _write_manifest(self, manifest_path: Path, manifest: dict) -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
