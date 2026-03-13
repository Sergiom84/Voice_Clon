from __future__ import annotations

from array import array
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import wave

import numpy as np
import soundfile as sf

from .settings import DEFAULT_FRAGMENT_PAUSE_MS, DEFAULT_OUTPUT_SAMPLE_RATE


@dataclass(frozen=True)
class ReferenceExcerpt:
    path: Path
    original_duration_seconds: float
    excerpt_duration_seconds: float
    start_seconds: float


def concatenate_fragments(
    fragments: list[Iterable[Any]],
    sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE,
    pause_ms: int = DEFAULT_FRAGMENT_PAUSE_MS,
) -> list[float]:
    pause_samples = int(sample_rate * pause_ms / 1000)
    pause = [0.0] * pause_samples
    combined: list[float] = []

    for index, fragment in enumerate(fragments):
        combined.extend(_coerce_audio_samples(fragment))
        if index < len(fragments) - 1:
            combined.extend(pause)

    return combined


def write_wav_file(path: str | Path, samples: Iterable[Any], sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    pcm = array("h", (_float_to_pcm16(sample) for sample in _coerce_audio_samples(samples)))
    with wave.open(str(target), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def assemble_wav_files(
    segment_paths: list[str | Path],
    target_path: str | Path,
    sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE,
    pause_ms: int = DEFAULT_FRAGMENT_PAUSE_MS,
) -> None:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    pause_samples = int(sample_rate * pause_ms / 1000)
    pause_frames = array("h", [0] * pause_samples).tobytes()

    with wave.open(str(target), "wb") as output_file:
        output_file.setnchannels(1)
        output_file.setsampwidth(2)
        output_file.setframerate(sample_rate)

        for index, segment_path in enumerate(segment_paths):
            with wave.open(str(segment_path), "rb") as segment_file:
                if segment_file.getnchannels() != 1:
                    raise ValueError(f"El segmento {segment_path} no es mono.")
                if segment_file.getsampwidth() != 2:
                    raise ValueError(f"El segmento {segment_path} no esta en PCM16.")
                if segment_file.getframerate() != sample_rate:
                    raise ValueError(
                        f"El segmento {segment_path} usa {segment_file.getframerate()} Hz y se esperaba {sample_rate} Hz."
                    )

                while True:
                    frames = segment_file.readframes(8192)
                    if not frames:
                        break
                    output_file.writeframesraw(frames)

            if index < len(segment_paths) - 1 and pause_frames:
                output_file.writeframesraw(pause_frames)


def extract_reference_excerpt(
    source_path: str | Path,
    target_path: str | Path,
    excerpt_seconds: float,
) -> ReferenceExcerpt:
    source = Path(source_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        audio_data, sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
    except Exception as exc:
        raise ValueError("No se pudo leer el audio de referencia para preparar el recorte.") from exc

    frame_count = int(audio_data.shape[0])

    if sample_rate <= 0 or frame_count <= 0:
        raise ValueError("El audio de referencia no contiene muestras validas.")

    original_duration_seconds = frame_count / float(sample_rate)
    excerpt_frame_count = max(1, min(int(excerpt_seconds * sample_rate), frame_count))
    start_frame = _find_loudest_excerpt_start_frame(
        audio_data=audio_data,
        sample_rate=sample_rate,
        excerpt_frame_count=excerpt_frame_count,
    )
    excerpt_data = audio_data[start_frame : start_frame + excerpt_frame_count]
    sf.write(str(target), excerpt_data, sample_rate, subtype="PCM_16")

    return ReferenceExcerpt(
        path=target,
        original_duration_seconds=original_duration_seconds,
        excerpt_duration_seconds=excerpt_frame_count / float(sample_rate),
        start_seconds=start_frame / float(sample_rate),
    )


def extract_reference_excerpts(
    source_path: str | Path,
    target_dir: str | Path,
    excerpt_seconds: float,
    excerpt_count: int,
) -> list[ReferenceExcerpt]:
    source = Path(source_path)
    target_root = Path(target_dir)
    target_root.mkdir(parents=True, exist_ok=True)

    try:
        audio_data, sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
    except Exception as exc:
        raise ValueError("No se pudo leer el audio de referencia para preparar los recortes.") from exc

    frame_count = int(audio_data.shape[0])
    if sample_rate <= 0 or frame_count <= 0:
        raise ValueError("El audio de referencia no contiene muestras validas.")

    excerpt_frame_count = max(1, int(excerpt_seconds * sample_rate))
    original_duration_seconds = frame_count / float(sample_rate)
    excerpts: list[ReferenceExcerpt] = []

    for index in range(excerpt_count):
        range_start = int(frame_count * index / excerpt_count)
        range_end = int(frame_count * (index + 1) / excerpt_count)
        if range_end <= range_start:
            continue

        scoped_excerpt_frame_count = min(excerpt_frame_count, range_end - range_start)
        start_frame = _find_loudest_excerpt_start_frame(
            audio_data=audio_data,
            sample_rate=sample_rate,
            excerpt_frame_count=scoped_excerpt_frame_count,
            range_start_frame=range_start,
            range_end_frame=range_end,
        )
        excerpt_data = audio_data[start_frame : start_frame + scoped_excerpt_frame_count]
        target_path = target_root / f"reference_{index + 1:02d}.wav"
        sf.write(str(target_path), excerpt_data, sample_rate, subtype="PCM_16")
        excerpts.append(
            ReferenceExcerpt(
                path=target_path,
                original_duration_seconds=original_duration_seconds,
                excerpt_duration_seconds=scoped_excerpt_frame_count / float(sample_rate),
                start_seconds=start_frame / float(sample_rate),
            )
        )

    return excerpts


def _coerce_audio_samples(fragment: Iterable[Any]) -> list[float]:
    if hasattr(fragment, "detach") and hasattr(fragment, "cpu"):
        fragment = fragment.detach().cpu()
    if hasattr(fragment, "tolist"):
        fragment = fragment.tolist()

    if isinstance(fragment, (bytes, bytearray, str)):
        raise TypeError("No se puede convertir el fragmento de audio a muestras float.")

    if not isinstance(fragment, Iterable):
        return [float(fragment)]

    flattened: list[float] = []
    for item in fragment:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, Iterable) and not isinstance(item, (bytes, bytearray, str)):
            flattened.extend(_coerce_audio_samples(item))
        else:
            flattened.append(float(item))
    return flattened


def _float_to_pcm16(sample: Any) -> int:
    value = float(sample)
    if value > 1.0:
        value = 1.0
    elif value < -1.0:
        value = -1.0
    return int(value * 32767.0)


def _find_loudest_excerpt_start_frame(
    audio_data: np.ndarray,
    sample_rate: int,
    excerpt_frame_count: int,
    range_start_frame: int = 0,
    range_end_frame: int | None = None,
) -> int:
    frame_count = int(audio_data.shape[0])
    if range_end_frame is None or range_end_frame > frame_count:
        range_end_frame = frame_count
    if range_start_frame < 0:
        range_start_frame = 0
    if range_start_frame >= range_end_frame:
        return 0
    if excerpt_frame_count >= range_end_frame - range_start_frame:
        return range_start_frame

    if audio_data.ndim == 1:
        mono_audio = audio_data
    else:
        mono_audio = np.mean(audio_data, axis=1)

    step_frames = max(sample_rate, excerpt_frame_count // 6, 1)
    best_start_frame = range_start_frame
    best_score = -1.0

    max_start = range_end_frame - excerpt_frame_count
    candidate_starts = list(range(range_start_frame, max_start + 1, step_frames))
    if candidate_starts[-1] != max_start:
        candidate_starts.append(max_start)

    for start_frame in candidate_starts:
        window = mono_audio[start_frame : start_frame + excerpt_frame_count]
        if window.size == 0:
            score = 0.0
        else:
            score = float(np.sqrt(np.mean(window * window)))
        if score > best_score:
            best_score = score
            best_start_frame = start_frame

    return best_start_frame
