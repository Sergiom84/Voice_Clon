from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def compute_speaker_similarity(reference_audio_path: str | Path, candidate_audio_path: str | Path) -> float | None:
    try:
        reference_embedding = compute_speaker_embedding(reference_audio_path)
        candidate_embedding = compute_speaker_embedding(candidate_audio_path)
    except Exception:
        return None
    return cosine_similarity(reference_embedding, candidate_embedding)


def compute_speaker_embedding(audio_path: str | Path, target_sample_rate: int = 16_000) -> np.ndarray:
    audio_data, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    if audio_data.shape[0] <= 0 or sample_rate <= 0:
        raise ValueError("Audio inválido para embedding de speaker.")

    mono = audio_data.mean(axis=1)
    mono = _normalize_signal(mono)
    mono = _resample_linear(mono, sample_rate, target_sample_rate)
    if mono.size <= target_sample_rate // 2:
        mono = np.pad(mono, (0, max(0, target_sample_rate // 2 - mono.size)))

    frame_size = int(0.025 * target_sample_rate)
    hop_size = int(0.010 * target_sample_rate)
    frames = _frame_signal(mono, frame_size=frame_size, hop_size=hop_size)
    if frames.size <= 0:
        raise ValueError("No se pudieron extraer frames del audio.")

    window = np.hanning(frame_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-8
    power = spectrum * spectrum
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(target_sample_rate))

    band_edges = [0, 250, 500, 1_000, 2_000, 3_500, 5_000, 6_500, 8_000]
    band_features: list[float] = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            band_energy = np.zeros((frames.shape[0],), dtype=np.float32)
        else:
            band_energy = np.mean(power[:, mask], axis=1)
        band_energy = np.log1p(np.maximum(0.0, band_energy))
        band_features.append(float(np.mean(band_energy)))
        band_features.append(float(np.std(band_energy)))

    spectral_centroid = np.sum(freqs[None, :] * power, axis=1) / np.maximum(np.sum(power, axis=1), 1e-8)
    spectral_rolloff = _spectral_rolloff(power=power, freqs=freqs, rolloff=0.85)
    zero_crossing_rate = _zero_crossing_rate(frames)
    rms = np.sqrt(np.mean(frames * frames, axis=1))

    stats = [
        float(np.mean(spectral_centroid)),
        float(np.std(spectral_centroid)),
        float(np.mean(spectral_rolloff)),
        float(np.std(spectral_rolloff)),
        float(np.mean(zero_crossing_rate)),
        float(np.std(zero_crossing_rate)),
        float(np.mean(rms)),
        float(np.std(rms)),
    ]
    vector = np.array([*band_features, *stats], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    value = float(np.dot(left, right) / (left_norm * right_norm))
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    if signal.size <= 0:
        return signal.astype(np.float32)
    peak = float(np.max(np.abs(signal)))
    if peak <= 1e-8:
        return signal.astype(np.float32)
    return (signal / peak).astype(np.float32)


def _resample_linear(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or signal.size <= 1:
        return signal.astype(np.float32)
    duration = float(signal.size - 1) / float(source_rate)
    target_size = max(2, int(duration * target_rate) + 1)
    source_times = np.linspace(0.0, duration, num=signal.size, dtype=np.float32)
    target_times = np.linspace(0.0, duration, num=target_size, dtype=np.float32)
    return np.interp(target_times, source_times, signal).astype(np.float32)


def _frame_signal(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if frame_size <= 0 or hop_size <= 0 or signal.size <= 0:
        return np.empty((0, frame_size), dtype=np.float32)
    if signal.size < frame_size:
        signal = np.pad(signal, (0, frame_size - signal.size))
    total_frames = 1 + (signal.size - frame_size) // hop_size
    frame_starts = np.arange(total_frames) * hop_size
    indices = frame_starts[:, None] + np.arange(frame_size)[None, :]
    return signal[indices].astype(np.float32)


def _spectral_rolloff(power: np.ndarray, freqs: np.ndarray, rolloff: float = 0.85) -> np.ndarray:
    cumulative = np.cumsum(power, axis=1)
    totals = cumulative[:, -1]
    thresholds = totals * rolloff
    indices = np.argmax(cumulative >= thresholds[:, None], axis=1)
    return freqs[indices]


def _zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    signs = np.sign(frames)
    signs[signs == 0] = 1
    crossings = np.sum(signs[:, 1:] != signs[:, :-1], axis=1)
    return crossings / np.maximum(1, frames.shape[1] - 1)
