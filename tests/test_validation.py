from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtts_spanish_app.validation import ValidationError, validate_reference_wav, validate_spanish_text


def _create_wav(path: Path, duration_seconds: float, sample_rate: int = 24_000) -> None:
    frame_count = int(duration_seconds * sample_rate)
    silence = b"\x00\x00" * frame_count
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence)


class ValidationTests(unittest.TestCase):
    def test_validate_spanish_text_rejects_empty_text(self) -> None:
        with self.assertRaises(ValidationError):
            validate_spanish_text("   \n  ")

    def test_validate_reference_wav_rejects_too_short_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "short.wav"
            _create_wav(wav_path, duration_seconds=1.5)
            with self.assertRaises(ValidationError):
                validate_reference_wav(str(wav_path))

    def test_validate_reference_wav_accepts_valid_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "valid.wav"
            _create_wav(wav_path, duration_seconds=5.0)
            duration = validate_reference_wav(str(wav_path))
            self.assertAlmostEqual(duration, 5.0, places=2)

    def test_validate_reference_wav_accepts_long_audio_for_auto_trim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "long.wav"
            _create_wav(wav_path, duration_seconds=199.0)
            duration = validate_reference_wav(str(wav_path))
            self.assertAlmostEqual(duration, 199.0, places=2)

    def test_validate_reference_wav_accepts_mp3_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mp3_path = Path(temp_dir) / "sample.mp3"
            mp3_path.write_bytes(b"fake")

            fake_metadata = SimpleNamespace(samplerate=24_000, frames=96_000)
            with mock.patch("xtts_spanish_app.validation.sf.info", return_value=fake_metadata):
                duration = validate_reference_wav(str(mp3_path))

            self.assertAlmostEqual(duration, 4.0, places=2)


if __name__ == "__main__":
    unittest.main()
