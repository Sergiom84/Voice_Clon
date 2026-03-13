from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .settings import (
    LONG_PROFILE_CONSOLIDATED_SECONDS,
    LONG_PROFILE_SEGMENT_COUNT,
    REFERENCE_PROFILE_AUTO,
    REFERENCE_PROFILE_LONG,
    REFERENCE_PROFILE_SHORT,
    SHORT_PROFILE_CONSOLIDATED_SECONDS,
    SHORT_PROFILE_SEGMENT_COUNT,
)


@dataclass(frozen=True)
class PreparedReference:
    profile: str
    path: Path
    source_duration_seconds: float
    output_duration_seconds: float
    windows: list[tuple[float, float]]

    @property
    def note(self) -> str:
        windows_text = ", ".join(f"{start:.1f}-{end:.1f}" for start, end in self.windows)
        return (
            f"Perfil de referencia aplicado: {self.profile}. "
            f"Audio original: {self.source_duration_seconds:.1f} s. "
            f"Referencia consolidada: {self.output_duration_seconds:.1f} s "
            f"(ventanas: {windows_text})."
        )


def resolve_reference_profile(requested_profile: str, duration_seconds: float) -> str:
    requested = (requested_profile or REFERENCE_PROFILE_AUTO).strip().lower()
    if requested in (REFERENCE_PROFILE_SHORT, REFERENCE_PROFILE_LONG):
        return requested
    if duration_seconds >= 300.0:
        return REFERENCE_PROFILE_LONG
    return REFERENCE_PROFILE_SHORT


def prepare_consolidated_reference(
    source_path: str | Path,
    target_path: str | Path,
    requested_profile: str,
) -> PreparedReference:
    source = Path(source_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    audio_data, sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
    if audio_data.shape[0] <= 0 or sample_rate <= 0:
        raise ValueError("El audio de referencia no contiene muestras válidas.")

    mono = audio_data.mean(axis=1)
    source_duration_seconds = float(mono.shape[0]) / float(sample_rate)
    profile = resolve_reference_profile(requested_profile, source_duration_seconds)

    if profile == REFERENCE_PROFILE_LONG:
        merged, windows = _build_long_profile_reference(mono=mono, sample_rate=sample_rate)
    else:
        merged, windows = _build_short_profile_reference(mono=mono, sample_rate=sample_rate)

    if merged.size <= 0:
        raise ValueError("No se pudo construir una referencia consolidada válida.")

    sf.write(str(target), merged.reshape(-1, 1), sample_rate, subtype="PCM_16")
    output_duration_seconds = float(merged.shape[0]) / float(sample_rate)

    return PreparedReference(
        profile=profile,
        path=target,
        source_duration_seconds=source_duration_seconds,
        output_duration_seconds=output_duration_seconds,
        windows=windows,
    )


def _build_short_profile_reference(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    segment_seconds = SHORT_PROFILE_CONSOLIDATED_SECONDS / float(SHORT_PROFILE_SEGMENT_COUNT)
    segment_frames = max(1, int(segment_seconds * sample_rate))
    starts = _pick_top_non_overlapping_windows(
        mono=mono,
        sample_rate=sample_rate,
        window_frames=segment_frames,
        picks=SHORT_PROFILE_SEGMENT_COUNT,
    )
    return _concatenate_windows(mono, sample_rate, starts, segment_frames)


def _build_long_profile_reference(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    segment_seconds = LONG_PROFILE_CONSOLIDATED_SECONDS / float(LONG_PROFILE_SEGMENT_COUNT)
    segment_frames = max(1, int(segment_seconds * sample_rate))
    frame_count = mono.shape[0]
    thirds = [
        (0, max(1, frame_count // 3)),
        (max(0, frame_count // 3), max(1, (2 * frame_count) // 3)),
        (max(0, (2 * frame_count) // 3), frame_count),
    ]
    starts: list[int] = []
    for start, end in thirds:
        starts.append(
            _find_best_window_in_range(
                mono=mono,
                sample_rate=sample_rate,
                window_frames=segment_frames,
                start_frame=start,
                end_frame=end,
            )
        )
    return _concatenate_windows(mono, sample_rate, starts, segment_frames)


def _concatenate_windows(
    mono: np.ndarray,
    sample_rate: int,
    starts: list[int],
    window_frames: int,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    chunks: list[np.ndarray] = []
    windows: list[tuple[float, float]] = []
    frame_count = mono.shape[0]
    for start in starts:
        bounded_start = min(max(0, start), max(0, frame_count - 1))
        end = min(frame_count, bounded_start + window_frames)
        chunk = mono[bounded_start:end]
        if chunk.size <= 0:
            continue
        chunks.append(chunk)
        windows.append((bounded_start / float(sample_rate), end / float(sample_rate)))
    if not chunks:
        return np.array([], dtype=np.float32), []
    return np.concatenate(chunks).astype(np.float32), windows


def _pick_top_non_overlapping_windows(
    mono: np.ndarray,
    sample_rate: int,
    window_frames: int,
    picks: int,
) -> list[int]:
    frame_count = mono.shape[0]
    if window_frames >= frame_count:
        return [0]

    step = max(sample_rate // 2, 1)
    candidates: list[tuple[float, int]] = []
    max_start = frame_count - window_frames
    for start in range(0, max_start + 1, step):
        candidates.append((_window_score(mono[start : start + window_frames]), start))
    if candidates[-1][1] != max_start:
        candidates.append((_window_score(mono[max_start : max_start + window_frames]), max_start))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[int] = []
    for _, start in candidates:
        if all(abs(start - existing) >= window_frames for existing in selected):
            selected.append(start)
        if len(selected) >= picks:
            break
    if not selected:
        selected.append(0)
    selected.sort()
    return selected


def _find_best_window_in_range(
    mono: np.ndarray,
    sample_rate: int,
    window_frames: int,
    start_frame: int,
    end_frame: int,
) -> int:
    if end_frame <= start_frame:
        return max(0, start_frame)

    scoped_length = end_frame - start_frame
    if scoped_length <= window_frames:
        return start_frame

    step = max(sample_rate // 2, 1)
    best_score = -1.0
    best_start = start_frame
    max_start = end_frame - window_frames
    for start in range(start_frame, max_start + 1, step):
        score = _window_score(mono[start : start + window_frames])
        if score > best_score:
            best_score = score
            best_start = start
    if best_start != max_start:
        tail_score = _window_score(mono[max_start : max_start + window_frames])
        if tail_score > best_score:
            best_start = max_start
    return best_start


def _window_score(chunk: np.ndarray) -> float:
    if chunk.size <= 0:
        return 0.0
    abs_chunk = np.abs(chunk)
    voiced_ratio = float(np.mean(abs_chunk > 0.02))
    rms = float(np.sqrt(np.mean(chunk * chunk)))
    dynamic = float(np.var(chunk))
    return (0.55 * voiced_ratio) + (0.3 * rms) + (0.15 * dynamic)
