"""Módulo de detección de problemas en segmentos de audio generados."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .settings import DEFAULT_OUTPUT_SAMPLE_RATE


@dataclass(frozen=True)
class SegmentQualityReport:
    """Resultado del análisis de calidad de un segmento."""

    is_valid: bool
    issues: list[str]
    duration_seconds: float
    expected_duration_seconds: float | None
    silence_ratio: float
    energy_variance: float


def analyze_segment_quality(
    audio_path: str | Path,
    text: str,
    sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE,
    min_chars_per_second: float = 5.0,
    max_chars_per_second: float = 25.0,
    max_silence_ratio: float = 0.4,
    min_energy_variance: float = 0.001,
) -> SegmentQualityReport:
    """
    Analiza la calidad de un segmento de audio generado.

    Args:
        audio_path: Ruta al archivo de audio WAV.
        text: Texto que se sintetizó.
        sample_rate: Tasa de muestreo esperada.
        min_chars_per_second: Mínimo de caracteres por segundo (demasiado lento).
        max_chars_per_second: Máximo de caracteres por segundo (demasiado rápido).
        max_silence_ratio: Máxima proporción de silencio permitida.
        min_energy_variance: Mínima varianza de energía (audio muy plano = problema).

    Returns:
        SegmentQualityReport con el análisis de calidad.
    """
    audio_path = Path(audio_path)
    issues: list[str] = []

    if not audio_path.exists():
        return SegmentQualityReport(
            is_valid=False,
            issues=[f"El archivo de audio no existe: {audio_path}"],
            duration_seconds=0.0,
            expected_duration_seconds=None,
            silence_ratio=1.0,
            energy_variance=0.0,
        )

    try:
        audio_data, sr = sf.read(str(audio_path))
    except Exception as exc:
        return SegmentQualityReport(
            is_valid=False,
            issues=[f"Error al leer audio: {exc}"],
            duration_seconds=0.0,
            expected_duration_seconds=None,
            silence_ratio=1.0,
            energy_variance=0.0,
        )

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    duration_seconds = len(audio_data) / sr
    char_count = len(text.strip())

    expected_duration_seconds = None
    if char_count > 0:
        expected_duration_seconds = char_count / ((min_chars_per_second + max_chars_per_second) / 2)

    silence_ratio = _calculate_silence_ratio(audio_data, sample_rate)
    energy_variance = float(np.var(audio_data))

    if char_count > 0 and duration_seconds > 0:
        chars_per_second = char_count / duration_seconds
        if chars_per_second < min_chars_per_second:
            issues.append(
                f"Audio demasiado lento ({chars_per_second:.1f} chars/seg, mínimo {min_chars_per_second}). "
                "Posible pérdida de foco o problema de síntesis."
            )
        elif chars_per_second > max_chars_per_second:
            issues.append(
                f"Audio demasiado rápido ({chars_per_second:.1f} chars/seg, máximo {max_chars_per_second}). "
                "Posible truncado o problema de síntesis."
            )

    if silence_ratio > max_silence_ratio:
        issues.append(
            f"Proporción de silencio alta ({silence_ratio:.1%}, máximo {max_silence_ratio:.1%}). "
            "El segmento puede contener pausas anómalas."
        )

    if energy_variance < min_energy_variance and len(audio_data) > sample_rate:
        issues.append(
            f"Audio muy plano (varianza {energy_variance:.6f}, mínimo {min_energy_variance}). "
            "Posible problema de síntesis o audio corrupto."
        )

    if duration_seconds < 0.5 and char_count > 10:
        issues.append(
            f"Duración muy corta ({duration_seconds:.2f}s) para {char_count} caracteres. "
            "Posible fallo de síntesis."
        )

    is_valid = len(issues) == 0

    return SegmentQualityReport(
        is_valid=is_valid,
        issues=issues,
        duration_seconds=duration_seconds,
        expected_duration_seconds=expected_duration_seconds,
        silence_ratio=silence_ratio,
        energy_variance=energy_variance,
    )


def _calculate_silence_ratio(audio_data: np.ndarray[Any, Any], sample_rate: int, silence_threshold: float = 0.02) -> float:
    """
    Calcula la proporción de silencio en el audio.

    Args:
        audio_data: Array de muestras de audio.
        sample_rate: Tasa de muestreo.
        silence_threshold: Umbral de amplitud para considerar silencio.

    Returns:
        Proporción de muestras en silencio (0.0 a 1.0).
    """
    if len(audio_data) == 0:
        return 1.0

    abs_audio = np.abs(audio_data)
    silence_samples = np.sum(abs_audio < silence_threshold)
    return float(silence_samples) / len(audio_data)


def detect_anomalous_duration(
    audio_path: str | Path,
    text: str,
    sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE,
    tolerance_factor: float = 2.0,
) -> tuple[bool, str]:
    """
    Detecta si la duración del audio es anómala respecto al texto.

    Args:
        audio_path: Ruta al archivo de audio.
        text: Texto sintetizado.
        sample_rate: Tasa de muestreo.
        tolerance_factor: Factor de tolerancia para duración esperada.

    Returns:
        Tupla (es_anómalo, descripción_del_problema).
    """
    report = analyze_segment_quality(audio_path, text, sample_rate)

    if report.expected_duration_seconds is None:
        return False, ""

    min_expected = report.expected_duration_seconds / tolerance_factor
    max_expected = report.expected_duration_seconds * tolerance_factor

    if report.duration_seconds < min_expected:
        return True, f"Duración {report.duration_seconds:.2f}s es muy corta (esperado ~{report.expected_duration_seconds:.1f}s)"
    if report.duration_seconds > max_expected:
        return True, f"Duración {report.duration_seconds:.2f}s es muy larga (esperado ~{report.expected_duration_seconds:.1f}s)"

    return False, ""
