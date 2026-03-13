from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .audio import concatenate_fragments, extract_reference_excerpt, write_wav_file
from .backend import VoiceBackend
from .settings import (
    DEFAULT_OUTPUT_SAMPLE_RATE,
    DEFAULT_REFERENCE_EXCERPT_SECONDS,
    MAX_REFERENCE_SECONDS,
    OUTPUTS_DIR,
    PREPARED_REFERENCES_DIR,
    ensure_app_directories,
)
from .text_processing import split_text_for_tts
from .validation import ValidationError, validate_reference_wav, validate_spanish_text


class SpanishVoiceService:
    def __init__(
        self,
        backend: VoiceBackend,
        output_dir: Path | None = None,
        prepared_reference_dir: Path | None = None,
    ) -> None:
        self.backend = backend
        self.output_dir = output_dir or OUTPUTS_DIR
        self.prepared_reference_dir = prepared_reference_dir or PREPARED_REFERENCES_DIR
        self._last_operation_note: str | None = None
        ensure_app_directories()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_reference_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_spanish(self, reference_audio_path: str | None, text_es: str) -> str:
        prepared_reference_path = self._prepare_reference_audio(reference_audio_path)
        normalized_text = validate_spanish_text(text_es)
        fragments = split_text_for_tts(normalized_text)

        if not fragments:
            raise ValidationError("No se pudo obtener ningun fragmento de texto valido para la sintesis.")

        rendered_fragments: list[list[float]] = []
        for fragment in fragments:
            rendered_fragments.append(
                self.backend.synthesize_fragment(text=fragment, reference_audio_path=prepared_reference_path)
            )

        combined_audio = concatenate_fragments(rendered_fragments, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE)
        output_path = self._build_output_path()
        write_wav_file(output_path, combined_audio, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE)
        return str(output_path)

    def describe_runtime(self) -> str:
        return self.backend.describe_runtime()

    def get_last_operation_note(self) -> str | None:
        return self._last_operation_note

    def _build_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.output_dir / f"voz_clonada_{timestamp}.wav"

    def _build_prepared_reference_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.prepared_reference_dir / f"referencia_preparada_{timestamp}.wav"

    def _prepare_reference_audio(self, reference_audio_path: str | None) -> str:
        duration_seconds = validate_reference_wav(reference_audio_path)
        self._last_operation_note = None

        if duration_seconds <= MAX_REFERENCE_SECONDS:
            return str(reference_audio_path)

        prepared_reference_path = self._build_prepared_reference_path()
        excerpt = extract_reference_excerpt(
            source_path=reference_audio_path,
            target_path=prepared_reference_path,
            excerpt_seconds=DEFAULT_REFERENCE_EXCERPT_SECONDS,
        )
        self._last_operation_note = (
            f"El audio de referencia duraba {excerpt.original_duration_seconds:.1f} s. "
            f"Se ha usado automaticamente un tramo de {excerpt.excerpt_duration_seconds:.1f} s "
            f"desde el segundo {excerpt.start_seconds:.1f}."
        )
        return str(excerpt.path)
