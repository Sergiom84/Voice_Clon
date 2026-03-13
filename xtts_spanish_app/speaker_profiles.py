from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .reference_profiles import prepare_consolidated_reference
from .settings import (
    REFERENCE_PROFILE_AUTO,
    SPEAKER_ADAPTATION_IDEAL_MAX_SECONDS,
    SPEAKER_ADAPTATION_IDEAL_MIN_SECONDS,
    SPEAKER_ADAPTATION_MIN_SECONDS,
    SPEAKER_PROFILES_DIR,
)


@dataclass(frozen=True)
class SpeakerProfile:
    name: str
    profile_dir: Path
    reference_audio_path: Path
    source_duration_seconds: float
    adaptation_ready: bool
    recommendation: str
    dataset_manifest_path: Path


class SpeakerProfileManager:
    def __init__(self, profiles_dir: Path | None = None) -> None:
        self.profiles_dir = profiles_dir or SPEAKER_PROFILES_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> list[str]:
        names = [entry.name for entry in self.profiles_dir.iterdir() if entry.is_dir() and (entry / "manifest.json").exists()]
        names.sort()
        return names

    def resolve_reference_path(self, profile_name: str) -> Path | None:
        normalized = normalize_speaker_profile_name(profile_name)
        if not normalized:
            return None
        profile_dir = self.profiles_dir / normalized
        manifest_path = profile_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        reference_path = Path(manifest.get("reference_audio_path", ""))
        if not reference_path.exists():
            return None
        return reference_path

    def create_or_update_profile(
        self,
        profile_name: str,
        source_audio_path: str | Path,
        requested_reference_profile: str = REFERENCE_PROFILE_AUTO,
    ) -> SpeakerProfile:
        normalized = normalize_speaker_profile_name(profile_name)
        if not normalized:
            raise ValueError("Nombre de perfil de speaker inválido.")

        profile_dir = self.profiles_dir / normalized
        profile_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = profile_dir / "dataset_segments"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        reference_path = profile_dir / "reference.wav"

        prepared_reference = prepare_consolidated_reference(
            source_path=source_audio_path,
            target_path=reference_path,
            requested_profile=requested_reference_profile,
        )

        dataset_manifest_path = profile_dir / "dataset_manifest.json"
        dataset_manifest = self._prepare_dataset(
            source_audio_path=source_audio_path,
            target_dir=dataset_dir,
        )
        dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        recommendation = _build_adaptation_recommendation(prepared_reference.source_duration_seconds)
        adaptation_ready = prepared_reference.source_duration_seconds >= SPEAKER_ADAPTATION_MIN_SECONDS

        manifest = {
            "name": normalized,
            "updated_at": datetime.now().isoformat(),
            "reference_audio_path": str(reference_path),
            "reference_profile": prepared_reference.profile,
            "source_duration_seconds": prepared_reference.source_duration_seconds,
            "consolidated_duration_seconds": prepared_reference.output_duration_seconds,
            "adaptation_ready": adaptation_ready,
            "recommendation": recommendation,
            "dataset_manifest_path": str(dataset_manifest_path),
        }
        manifest_path = profile_dir / "manifest.json"
        if manifest_path.exists():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                previous = {}
            if "created_at" in previous:
                manifest["created_at"] = previous["created_at"]
        if "created_at" not in manifest:
            manifest["created_at"] = datetime.now().isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        return SpeakerProfile(
            name=normalized,
            profile_dir=profile_dir,
            reference_audio_path=reference_path,
            source_duration_seconds=prepared_reference.source_duration_seconds,
            adaptation_ready=adaptation_ready,
            recommendation=recommendation,
            dataset_manifest_path=dataset_manifest_path,
        )

    def _prepare_dataset(self, source_audio_path: str | Path, target_dir: Path) -> dict:
        source = Path(source_audio_path)
        audio_data, sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
        mono = audio_data.mean(axis=1)
        duration_seconds = float(mono.shape[0]) / float(sample_rate) if sample_rate > 0 else 0.0

        segments = _segment_by_silence(mono=mono, sample_rate=sample_rate)
        segment_payload: list[dict] = []
        for index, (start_frame, end_frame) in enumerate(segments, start=1):
            chunk = mono[start_frame:end_frame]
            if chunk.size <= 0:
                continue
            segment_path = target_dir / f"segment_{index:04d}.wav"
            sf.write(str(segment_path), chunk.reshape(-1, 1), sample_rate, subtype="PCM_16")
            segment_payload.append(
                {
                    "index": index,
                    "path": str(segment_path),
                    "start_seconds": start_frame / float(sample_rate),
                    "end_seconds": end_frame / float(sample_rate),
                    "duration_seconds": (end_frame - start_frame) / float(sample_rate),
                    "transcript": "",
                }
            )

        return {
            "source_audio_path": str(source),
            "source_duration_seconds": duration_seconds,
            "transcription_status": "pending",
            "notes": (
                "Pipeline listo para adaptación: segmentos preparados. "
                "Completa transcripciones y conecta entrenamiento de speaker cuando se habilite."
            ),
            "segments": segment_payload,
        }


def normalize_speaker_profile_name(profile_name: str | None) -> str | None:
    if not profile_name:
        return None
    cleaned = profile_name.strip().lower()
    if not cleaned:
        return None
    cleaned = re.sub(r"[^a-z0-9_-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return None
    return cleaned


def _segment_by_silence(
    mono: np.ndarray,
    sample_rate: int,
    silence_threshold: float = 0.018,
    min_silence_seconds: float = 0.35,
    min_segment_seconds: float = 2.0,
    max_segment_seconds: float = 12.0,
) -> list[tuple[int, int]]:
    if mono.size <= 0 or sample_rate <= 0:
        return []
    abs_audio = np.abs(mono)
    voiced_mask = abs_audio > silence_threshold
    min_silence_frames = max(1, int(min_silence_seconds * sample_rate))
    min_segment_frames = max(1, int(min_segment_seconds * sample_rate))
    max_segment_frames = max(min_segment_frames, int(max_segment_seconds * sample_rate))

    segments: list[tuple[int, int]] = []
    frame_count = mono.shape[0]
    cursor = 0
    while cursor < frame_count:
        while cursor < frame_count and not voiced_mask[cursor]:
            cursor += 1
        if cursor >= frame_count:
            break
        start = cursor
        silence_run = 0
        while cursor < frame_count:
            if voiced_mask[cursor]:
                silence_run = 0
            else:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    break
            if cursor - start >= max_segment_frames:
                break
            cursor += 1
        end = cursor - silence_run if silence_run >= min_silence_frames else min(cursor + 1, frame_count)
        if end - start >= min_segment_frames:
            segments.append((start, end))
        cursor = max(cursor, end)

    if not segments:
        return [(0, min(frame_count, max_segment_frames))]
    return segments


def _build_adaptation_recommendation(duration_seconds: float) -> str:
    if duration_seconds < SPEAKER_ADAPTATION_MIN_SECONDS:
        missing = int(max(0, SPEAKER_ADAPTATION_MIN_SECONDS - duration_seconds))
        return f"Aún no ideal para adaptación. Recomendado añadir ~{missing} s para alcanzar >=10 min útiles."
    if duration_seconds <= SPEAKER_ADAPTATION_IDEAL_MAX_SECONDS:
        return (
            "Rango recomendado para adaptación alcanzado. "
            f"Objetivo ideal: {int(SPEAKER_ADAPTATION_IDEAL_MIN_SECONDS // 60)}-{int(SPEAKER_ADAPTATION_IDEAL_MAX_SECONDS // 60)} min."
        )
    return "Duración alta detectada. Conviene depurar segmentos para mantener consistencia de timbre antes de entrenar."
