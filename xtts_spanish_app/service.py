from __future__ import annotations

import json
import shutil
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .audio import assemble_wav_files, write_wav_file
from .backend import VoiceBackend
from .quality_check import SegmentQualityReport, analyze_segment_quality
from .reference_profiles import PreparedReference, prepare_consolidated_reference
from .settings import (
    DEFAULT_OUTPUT_SAMPLE_RATE,
    DEFAULT_SYNTHESIS_SEED,
    FIDELITY_MODE_MAXIMUM,
    JOBS_DIR,
    MAX_FIDELITY_CANDIDATES,
    MAX_RESYNTHESIS_ATTEMPTS,
    SPEAKER_SIMILARITY_THRESHOLD,
    SPEAKER_SIMILARITY_WARNING_THRESHOLD,
    ensure_app_directories,
)
from .speaker_profiles import SpeakerProfileManager, normalize_speaker_profile_name
from .speaker_similarity import compute_speaker_similarity
from .text_processing import build_synthesis_chunks
from .validation import (
    ValidationError,
    validate_fidelity_mode,
    validate_reference_profile,
    validate_reference_wav,
    validate_spanish_text,
)


@dataclass(frozen=True)
class SynthesisResult:
    final_audio_path: str
    segment_count: int
    reference_paths_used: list[str]
    operation_note: str | None
    job_dir: str
    quality_issues: list[str] = field(default_factory=list)
    resynthesized_count: int = 0
    reference_profile: str = "auto"
    fidelity_mode: str = "maxima"
    speaker_similarity_avg: float | None = None
    speaker_similarity_min: float | None = None
    similarity_warnings: list[str] = field(default_factory=list)
    speaker_profile_used: str | None = None
    speaker_profile_saved: str | None = None
    speaker_profile_recommendation: str | None = None


@dataclass(frozen=True)
class SegmentSynthesisOutcome:
    attempts: int
    candidate_count: int
    selected_seed: int
    quality_issues: list[str] = field(default_factory=list)
    speaker_similarity: float | None = None
    similarity_warning: str | None = None


class SpanishVoiceService:
    def __init__(
        self,
        backend: VoiceBackend,
        jobs_dir: Path | None = None,
        speaker_profile_manager: SpeakerProfileManager | None = None,
        quality_analyzer: Callable[[str | Path, str], SegmentQualityReport] | None = None,
        speaker_similarity_fn: Callable[[str | Path, str | Path], float | None] | None = None,
    ) -> None:
        self.backend = backend
        self.jobs_dir = jobs_dir or JOBS_DIR
        self.speaker_profile_manager = speaker_profile_manager or SpeakerProfileManager()
        self.quality_analyzer = quality_analyzer or analyze_segment_quality
        self.speaker_similarity_fn = speaker_similarity_fn or compute_speaker_similarity
        self._last_operation_note: str | None = None
        ensure_app_directories()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_spanish(
        self,
        reference_audio_path: str | None,
        text_es: str,
        reference_profile: str = "auto",
        fidelity_mode: str = "maxima",
        speaker_profile_name: str | None = None,
        save_speaker_profile: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> SynthesisResult:
        normalized_text = validate_spanish_text(text_es)
        chunks = build_synthesis_chunks(normalized_text)
        normalized_profile = validate_reference_profile(reference_profile)
        normalized_mode = validate_fidelity_mode(fidelity_mode)
        normalized_speaker_profile = normalize_speaker_profile_name(speaker_profile_name)

        if not chunks:
            raise ValidationError("No se pudo obtener ningun fragmento de texto valido para la sintesis.")

        total_steps = len(chunks) + 2
        self._notify_progress(progress_callback, "Preparando referencia...", 0, total_steps)
        job_dir = self._build_job_dir()
        segments_dir = job_dir / "segments"
        references_dir = job_dir / "references"
        segments_dir.mkdir(parents=True, exist_ok=True)
        references_dir.mkdir(parents=True, exist_ok=True)

        source_reference_path, speaker_profile_used = self._resolve_reference_source(
            reference_audio_path=reference_audio_path,
            speaker_profile_name=normalized_speaker_profile,
        )
        prepared_reference = self._prepare_reference_audio(
            source_reference_path=source_reference_path,
            references_dir=references_dir,
            requested_profile=normalized_profile,
        )
        reference_paths = [str(prepared_reference.path)]
        reference_path = reference_paths[0]

        speaker_profile_saved = None
        speaker_profile_recommendation = None
        if save_speaker_profile and normalized_speaker_profile:
            if not reference_audio_path:
                raise ValidationError("Para guardar o actualizar un speaker profile debes subir un audio de referencia.")
            profile = self.speaker_profile_manager.create_or_update_profile(
                profile_name=normalized_speaker_profile,
                source_audio_path=reference_audio_path,
                requested_reference_profile=prepared_reference.profile,
            )
            speaker_profile_saved = profile.name
            speaker_profile_recommendation = profile.recommendation

        manifest_path = job_dir / "manifest.json"
        final_audio_path = job_dir / "final.wav"
        manifest = {
            "job_id": job_dir.name,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "job_dir": str(job_dir),
            "final_audio_path": str(final_audio_path),
            "reference_paths_used": reference_paths,
            "reference_profile": prepared_reference.profile,
            "fidelity_mode": normalized_mode,
            "speaker_profile_used": speaker_profile_used,
            "speaker_profile_saved": speaker_profile_saved,
            "speaker_profile_recommendation": speaker_profile_recommendation,
            "operation_note": self._last_operation_note,
            "segment_count": len(chunks),
            "segments": [
                {
                    "index": index,
                    "path": str(segments_dir / f"segment_{index:03d}.wav"),
                    "char_count": len(chunk),
                    "text_preview": chunk[:120],
                    "status": "pending",
                    "attempts": 0,
                    "candidate_count": 0,
                    "selected_seed": None,
                    "speaker_similarity": None,
                }
                for index, chunk in enumerate(chunks, start=1)
            ],
        }
        self._write_manifest(manifest_path, manifest)

        segment_paths: list[str] = []
        quality_issues: list[str] = []
        resynthesized_count = 0
        speaker_similarities: list[float] = []
        similarity_warnings: list[str] = []

        try:
            for index, chunk in enumerate(chunks, start=1):
                self._notify_progress(progress_callback, f"Sintetizando bloque {index}/{len(chunks)}...", index, total_steps)

                segment_path = segments_dir / f"segment_{index:03d}.wav"
                base_seed = DEFAULT_SYNTHESIS_SEED + index
                if normalized_mode == FIDELITY_MODE_MAXIMUM:
                    outcome = self._synthesize_with_fidelity_ranking(
                        text=chunk,
                        reference_path=reference_path,
                        reference_paths=reference_paths,
                        segment_path=segment_path,
                        base_seed=base_seed,
                    )
                else:
                    outcome = self._synthesize_with_quality_check(
                        text=chunk,
                        reference_path=reference_path,
                        reference_paths=reference_paths,
                        segment_path=segment_path,
                        base_seed=base_seed,
                    )

                if outcome.attempts > 1:
                    resynthesized_count += 1
                    quality_issues.append(f"Bloque {index}: re-sintetizado ({outcome.attempts} intentos)")

                for issue in outcome.quality_issues:
                    quality_issues.append(f"Bloque {index}: {issue}")

                if outcome.speaker_similarity is not None:
                    speaker_similarities.append(outcome.speaker_similarity)
                if outcome.similarity_warning:
                    similarity_warnings.append(f"Bloque {index}: {outcome.similarity_warning}")

                segment_paths.append(str(segment_path))
                manifest["segments"][index - 1]["status"] = "completed"
                manifest["segments"][index - 1]["attempts"] = outcome.attempts
                manifest["segments"][index - 1]["candidate_count"] = outcome.candidate_count
                manifest["segments"][index - 1]["selected_seed"] = outcome.selected_seed
                manifest["segments"][index - 1]["speaker_similarity"] = outcome.speaker_similarity
                self._write_manifest(manifest_path, manifest)

            self._notify_progress(progress_callback, "Ensamblando audio final...", len(chunks) + 1, total_steps)
            assemble_wav_files(segment_paths=segment_paths, target_path=final_audio_path, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE)
            manifest["status"] = "completed"
            manifest["completed_at"] = datetime.now().isoformat()
            manifest["quality_issues"] = quality_issues
            manifest["resynthesized_count"] = resynthesized_count
            manifest["speaker_similarity_avg"] = _safe_mean(speaker_similarities)
            manifest["speaker_similarity_min"] = min(speaker_similarities) if speaker_similarities else None
            manifest["similarity_warnings"] = similarity_warnings
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
            reference_profile=prepared_reference.profile,
            fidelity_mode=normalized_mode,
            speaker_similarity_avg=_safe_mean(speaker_similarities),
            speaker_similarity_min=min(speaker_similarities) if speaker_similarities else None,
            similarity_warnings=similarity_warnings,
            speaker_profile_used=speaker_profile_used,
            speaker_profile_saved=speaker_profile_saved,
            speaker_profile_recommendation=speaker_profile_recommendation,
        )

    def _synthesize_with_quality_check(
        self,
        text: str,
        reference_path: str,
        reference_paths: list[str],
        segment_path: Path,
        base_seed: int,
    ) -> SegmentSynthesisOutcome:
        all_issues: list[str] = []
        selected_seed = base_seed

        for attempt in range(1, MAX_RESYNTHESIS_ATTEMPTS + 1):
            seed = base_seed + (attempt - 1) * 1000
            selected_seed = seed
            rendered_audio = self.backend.synthesize_fragment(
                text=text,
                reference_audio_paths=reference_paths,
                seed=seed,
            )
            write_wav_file(segment_path, rendered_audio, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE)

            report = self.quality_analyzer(segment_path, text)
            similarity = self.speaker_similarity_fn(reference_path, segment_path)

            if report.is_valid:
                warning = None
                if similarity is not None and similarity < SPEAKER_SIMILARITY_THRESHOLD:
                    warning = (
                        f"Similitud de speaker baja ({similarity:.3f}). "
                        "Considera usar modo Máxima o mejorar el audio de referencia."
                    )
                return SegmentSynthesisOutcome(
                    attempts=attempt,
                    candidate_count=attempt,
                    selected_seed=seed,
                    quality_issues=all_issues,
                    speaker_similarity=similarity,
                    similarity_warning=warning,
                )

            all_issues.extend(report.issues)

            if attempt < MAX_RESYNTHESIS_ATTEMPTS:
                continue

        similarity = self.speaker_similarity_fn(reference_path, segment_path)
        warning = "No se alcanzó calidad válida tras los intentos disponibles."
        if similarity is not None and similarity < SPEAKER_SIMILARITY_WARNING_THRESHOLD:
            warning = f"{warning} Similitud de speaker baja ({similarity:.3f})."
        return SegmentSynthesisOutcome(
            attempts=MAX_RESYNTHESIS_ATTEMPTS,
            candidate_count=MAX_RESYNTHESIS_ATTEMPTS,
            selected_seed=selected_seed,
            quality_issues=all_issues,
            speaker_similarity=similarity,
            similarity_warning=warning,
        )

    def _synthesize_with_fidelity_ranking(
        self,
        text: str,
        reference_path: str,
        reference_paths: list[str],
        segment_path: Path,
        base_seed: int,
    ) -> SegmentSynthesisOutcome:
        candidate_dir = segment_path.parent / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[dict] = []

        for attempt in range(1, MAX_FIDELITY_CANDIDATES + 1):
            seed = base_seed + (attempt - 1) * 1000
            candidate_path = candidate_dir / f"{segment_path.stem}_candidate_{attempt:02d}.wav"
            rendered_audio = self.backend.synthesize_fragment(
                text=text,
                reference_audio_paths=reference_paths,
                seed=seed,
            )
            write_wav_file(candidate_path, rendered_audio, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE)
            quality_report = self.quality_analyzer(candidate_path, text)
            quality_score = _score_quality_report(quality_report)
            similarity = self.speaker_similarity_fn(reference_path, candidate_path)
            similarity_score = similarity if similarity is not None else 0.0
            final_score = (0.55 * quality_score) + (0.45 * similarity_score)
            candidates.append(
                {
                    "attempt": attempt,
                    "seed": seed,
                    "path": candidate_path,
                    "quality_report": quality_report,
                    "quality_score": quality_score,
                    "similarity": similarity,
                    "final_score": final_score,
                }
            )

        best = max(candidates, key=lambda candidate: candidate["final_score"])
        shutil.copy2(best["path"], segment_path)

        quality_issues = list(best["quality_report"].issues)
        similarity = best["similarity"]
        similarity_warning = None

        if similarity is None:
            similarity_warning = "No se pudo calcular similitud de speaker para este bloque."
        elif similarity < SPEAKER_SIMILARITY_WARNING_THRESHOLD:
            similarity_warning = (
                f"Similitud de speaker muy baja ({similarity:.3f} < {SPEAKER_SIMILARITY_WARNING_THRESHOLD:.2f})."
            )
        elif similarity < SPEAKER_SIMILARITY_THRESHOLD:
            similarity_warning = (
                f"Similitud de speaker por debajo del umbral ({similarity:.3f} < {SPEAKER_SIMILARITY_THRESHOLD:.2f}). "
                "Prueba una referencia con menos ruido para mejorar timbre."
            )

        invalid_candidates = [candidate for candidate in candidates if not candidate["quality_report"].is_valid]
        if len(invalid_candidates) == len(candidates):
            extra = "Todos los candidatos presentaron anomalías de calidad."
            quality_issues.append(extra)

        for candidate in candidates:
            if candidate["path"] != best["path"] and candidate["path"].exists():
                candidate["path"].unlink(missing_ok=True)

        return SegmentSynthesisOutcome(
            attempts=int(best["attempt"]),
            candidate_count=MAX_FIDELITY_CANDIDATES,
            selected_seed=int(best["seed"]),
            quality_issues=quality_issues,
            speaker_similarity=similarity,
            similarity_warning=similarity_warning,
        )

    def describe_runtime(self) -> str:
        return self.backend.describe_runtime()

    def get_last_operation_note(self) -> str | None:
        return self._last_operation_note

    def _build_job_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.jobs_dir / f"job_{timestamp}_{uuid4().hex[:8]}"

    def _resolve_reference_source(
        self,
        reference_audio_path: str | None,
        speaker_profile_name: str | None,
    ) -> tuple[str, str | None]:
        if reference_audio_path:
            validate_reference_wav(reference_audio_path)
            return reference_audio_path, None
        if speaker_profile_name:
            resolved = self.speaker_profile_manager.resolve_reference_path(speaker_profile_name)
            if not resolved:
                raise ValidationError(
                    f"No existe speaker profile '{speaker_profile_name}'. Sube un audio o crea antes ese perfil."
                )
            validate_reference_wav(str(resolved))
            return str(resolved), speaker_profile_name
        raise ValidationError("Sube un audio de referencia o indica un speaker profile existente.")

    def _prepare_reference_audio(
        self,
        source_reference_path: str,
        references_dir: Path,
        requested_profile: str,
    ) -> PreparedReference:
        target_path = references_dir / "reference_01.wav"
        prepared = prepare_consolidated_reference(
            source_path=source_reference_path,
            target_path=target_path,
            requested_profile=requested_profile,
        )
        self._last_operation_note = prepared.note
        return prepared

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


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _score_quality_report(report: SegmentQualityReport) -> float:
    score = 1.0
    if not report.is_valid:
        score -= 0.25 + min(0.35, 0.06 * len(report.issues))
    if report.silence_ratio > 0.35:
        score -= min(0.3, (report.silence_ratio - 0.35) * 1.5)
    if report.energy_variance < 0.001:
        score -= 0.2
    if report.duration_seconds < 0.6:
        score -= 0.2
    return max(0.0, min(1.0, score))
