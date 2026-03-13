from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from xtts_spanish_app.backend import VoiceBackend
from xtts_spanish_app.service import SpanishVoiceService


def _create_wav(path: Path, duration_seconds: float, sample_rate: int = 24_000) -> None:
    frame_count = int(duration_seconds * sample_rate)
    silence = b"\x00\x00" * frame_count
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence)


class FakeBackend(VoiceBackend):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.reference_paths: list[str] = []

    def load_model(self) -> None:
        return None

    def synthesize_fragment(self, text: str, reference_audio_path: str) -> list[float]:
        self.calls.append(text)
        self.reference_paths.append(reference_audio_path)
        return [0.0, 0.1, -0.1, 0.0]

    def describe_runtime(self) -> str:
        return "fake-backend"


class SpanishVoiceServiceTests(unittest.TestCase):
    def test_synthesize_spanish_writes_output_file(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            service = SpanishVoiceService(backend=backend, output_dir=Path(temp_dir))
            wav_path = Path(temp_dir) / "reference.wav"
            _create_wav(wav_path, duration_seconds=4.0)

            output_path = Path(service.synthesize_spanish(str(wav_path), "Hola. Esto es una prueba."))

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertEqual(backend.calls, ["Hola.", "Esto es una prueba."])

    def test_synthesize_spanish_extracts_excerpt_for_long_reference_audio(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            service = SpanishVoiceService(backend=backend, output_dir=Path(temp_dir), prepared_reference_dir=Path(temp_dir) / "prepared")
            wav_path = Path(temp_dir) / "long_reference.wav"
            _create_wav(wav_path, duration_seconds=45.0)

            service.synthesize_spanish(str(wav_path), "Hola.")

            used_reference = Path(backend.reference_paths[0])
            self.assertNotEqual(used_reference, wav_path)
            self.assertTrue(used_reference.exists())
            self.assertIn("Se ha usado automaticamente un tramo", service.get_last_operation_note())

            with wave.open(str(used_reference), "rb") as wav_file:
                duration_seconds = wav_file.getnframes() / float(wav_file.getframerate())

            self.assertAlmostEqual(duration_seconds, 20.0, places=1)


if __name__ == "__main__":
    unittest.main()
