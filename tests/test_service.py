from __future__ import annotations

import tempfile
import unittest
import wave
import json
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
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[str] = []
        self.reference_paths: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def load_model(self) -> None:
        return None

    def synthesize_fragment(self, text: str, reference_audio_paths: list[str]) -> list[float]:
        self.calls.append(text)
        self.reference_paths.append(reference_audio_paths)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("backend roto")
        return [0.0, 0.1, -0.1, 0.0]

    def describe_runtime(self) -> str:
        return "fake-backend"


class SpanishVoiceServiceTests(unittest.TestCase):
    def test_synthesize_spanish_writes_job_artifacts_for_short_reference(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            service = SpanishVoiceService(backend=backend, jobs_dir=Path(temp_dir))
            wav_path = Path(temp_dir) / "reference.wav"
            _create_wav(wav_path, duration_seconds=4.0)
            progress_events: list[str] = []

            result = service.synthesize_spanish(
                str(wav_path),
                "Hola. Esto es una prueba.",
                progress_callback=lambda status, step, total: progress_events.append(f"{step}/{total}:{status}"),
            )

            output_path = Path(result.final_audio_path)
            job_dir = Path(result.job_dir)
            manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertEqual(result.segment_count, 1)
            self.assertEqual(len(result.reference_paths_used), 1)
            self.assertEqual(backend.calls, ["Hola. Esto es una prueba."])
            self.assertEqual(len(backend.reference_paths[0]), 1)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["segment_count"], 1)
            self.assertTrue((job_dir / "segments" / "segment_001.wav").exists())
            self.assertTrue((job_dir / "references").exists())
            self.assertIn("0/3:Preparando referencia...", progress_events[0])
            self.assertIn("1/3:Sintetizando bloque 1/1...", progress_events[1])
            self.assertIn("2/3:Ensamblando audio final...", progress_events[2])
            self.assertIn("3/3:Listo.", progress_events[3])

    def test_synthesize_spanish_prepares_three_references_for_long_audio(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            service = SpanishVoiceService(backend=backend, jobs_dir=Path(temp_dir))
            wav_path = Path(temp_dir) / "long_reference.wav"
            _create_wav(wav_path, duration_seconds=180.0)

            result = service.synthesize_spanish(str(wav_path), "Hola.")

            self.assertEqual(result.segment_count, 1)
            self.assertEqual(len(result.reference_paths_used), 3)
            self.assertEqual(backend.reference_paths[0], result.reference_paths_used)
            self.assertIn("Se han usado automaticamente 3 tramos", result.operation_note or "")

            for used_reference in result.reference_paths_used:
                used_reference_path = Path(used_reference)
                self.assertNotEqual(used_reference_path, wav_path)
                self.assertTrue(used_reference_path.exists())
                with wave.open(str(used_reference_path), "rb") as wav_file:
                    duration_seconds = wav_file.getnframes() / float(wav_file.getframerate())
                self.assertAlmostEqual(duration_seconds, 10.0, places=1)

    def test_synthesize_spanish_keeps_partial_artifacts_on_failed_block(self) -> None:
        backend = FakeBackend(fail_on_call=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            service = SpanishVoiceService(backend=backend, jobs_dir=Path(temp_dir))
            wav_path = Path(temp_dir) / "reference.wav"
            _create_wav(wav_path, duration_seconds=4.0)
            text = " ".join(["palabra"] * 220)

            with self.assertRaises(RuntimeError) as context:
                service.synthesize_spanish(str(wav_path), text)

            self.assertIn("Fallo en bloque 2/", str(context.exception))
            job_dirs = list(Path(temp_dir).glob("job_*"))
            self.assertEqual(len(job_dirs), 1)
            job_dir = job_dirs[0]
            manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failed_segment"], 2)
            self.assertTrue((job_dir / "segments" / "segment_001.wav").exists())
            self.assertFalse((job_dir / "final.wav").exists())


if __name__ == "__main__":
    unittest.main()
