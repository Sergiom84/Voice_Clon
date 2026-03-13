from __future__ import annotations

from pathlib import Path
import re

import soundfile as sf

from .settings import (
    FIDELITY_MODE_OPTIONS,
    FIDELITY_MODE_MAXIMUM,
    MIN_REFERENCE_SECONDS,
    REFERENCE_PROFILE_AUTO,
    REFERENCE_PROFILE_OPTIONS,
)
from .text_processing import normalize_text


class ValidationError(ValueError):
    """Error controlado de validacion de la UI."""


def validate_reference_wav(reference_audio_path: str | None) -> float:
    if not reference_audio_path:
        raise ValidationError("Sube un archivo WAV o MP3 de referencia antes de generar audio.")

    path = Path(reference_audio_path)
    if not path.exists():
        raise ValidationError("El archivo de referencia no existe o ya no esta disponible.")

    try:
        metadata = sf.info(str(path))
        frame_rate = int(metadata.samplerate)
        frame_count = int(metadata.frames)
    except Exception as exc:
        raise ValidationError("El audio de referencia debe ser un WAV o MP3 valido.") from exc

    if frame_rate <= 0:
        raise ValidationError("No se pudo leer la frecuencia de muestreo del WAV de referencia.")

    duration_seconds = frame_count / float(frame_rate)

    if duration_seconds < MIN_REFERENCE_SECONDS:
        raise ValidationError(
            f"El audio de referencia es demasiado corto. Usa un WAV o MP3 de al menos {MIN_REFERENCE_SECONDS:.0f} segundos."
        )

    return duration_seconds


def validate_spanish_text(text_es: str) -> str:
    normalized = normalize_text(text_es)
    if not normalized:
        raise ValidationError("Escribe el texto en espanol que quieres sintetizar.")
    return text_es


def validate_reference_profile(reference_profile: str | None) -> str:
    profile = (reference_profile or REFERENCE_PROFILE_AUTO).strip().lower()
    if profile not in REFERENCE_PROFILE_OPTIONS:
        valid = ", ".join(REFERENCE_PROFILE_OPTIONS)
        raise ValidationError(f"Perfil de referencia inválido. Usa: {valid}.")
    return profile


def validate_fidelity_mode(fidelity_mode: str | None) -> str:
    mode = (fidelity_mode or FIDELITY_MODE_MAXIMUM).strip().lower()
    if mode not in FIDELITY_MODE_OPTIONS:
        valid = ", ".join(FIDELITY_MODE_OPTIONS)
        raise ValidationError(f"Modo de fidelidad inválido. Usa: {valid}.")
    return mode


def normalize_profile_name(profile_name: str | None) -> str | None:
    if not profile_name:
        return None
    cleaned = profile_name.strip().lower()
    if not cleaned:
        return None
    cleaned = re.sub(r"[^a-z0-9_-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or None
