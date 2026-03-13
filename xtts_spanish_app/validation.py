from __future__ import annotations

from pathlib import Path

import soundfile as sf

from .settings import MAX_REFERENCE_SECONDS, MIN_REFERENCE_SECONDS
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
    return normalized
