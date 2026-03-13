from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

from xtts_spanish_app.backend import VoiceBackend
from xtts_spanish_app.quality_check import SegmentQualityReport
from xtts_spanish_app.service import SpanishVoiceService
from xtts_spanish_app.speaker_profiles import SpeakerProfileManager


def _create_wav(path: Path, duration_seconds: float, sample_rate: int = 24_000, amplitude: float = 0.15) -> None:
    import math

    frame_count = int(duration_seconds * sample_rate)
    samples = bytearray()
    for index in range(frame_count):
        value = amplitude * math.sin(2.0 * math.pi * 220.0 * (index / float(sample_rate)))
        pcm16 = int(max(-1.0, min(1.0, value)) * 32767.0)
        samples.extend(int(pcm16).to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(samples))


class FakeBackend(VoiceBackend):
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[str] = []
        self.reference_paths: list[list[str]] = []
        self.seeds: list[int | None] = []
        self.fail_on_call = fail_on_call

    def load_model(self) -> None:
        return None

    def synthesize_fragment(
        self,
        text: str,
        reference_audio_paths: list[str],
        seed: int | None = None,
    ) -> list[float]:
        self.calls.append(text)
        self.reference_paths.append(reference_audio_paths)
        self.seeds.append(seed)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("backend roto")
        return [0.0, 0.1, -0.1, 0.0] * 1_200

    def describe_runtime(self) -> str:
        return "fake-backend"


def _always_valid_quality(_audio_path: str | Path, text: str) -> SegmentQualityReport:
    return SegmentQualityReport(
        is_valid=True,
        issues=[],
        duration_seconds=max(0.7, len(text) / 18.0),
        expected_duration_seconds=max(0.7, len(text) / 18.0),
        silence_ratio=0.1,
        energy_variance=0.01,
    )


class SpanishVoiceServiceTests(unittest.TestCase):
    def test_synthesize_spanish_writes_manifest_with_new_fields(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir) / "profiles"
            service = SpanishVoiceService(
                backend=backend,
                jobs_dir=Path(temp_dir) / "jobs",
                speaker_profile_manager=SpeakerProfileManager(profiles_dir=profiles_dir),
                quality_analyzer=_always_valid_quality,
                speaker_similarity_fn=lambda *_args, **_kwargs: 0.83,
            )
            wav_path = Path(temp_dir) / "reference.wav"
            _create_wav(wav_path, duration_seconds=8.0)

            result = service.synthesize_spanish(
                str(wav_path),
                "Hola. Esto es una prueba.",
                reference_profile="auto",
                fidelity_mode="normal",
            )

            job_dir = Path(result.job_dir)
            manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["reference_profile"], "short")
            self.assertEqual(manifest["fidelity_mode"], "normal")
            self.assertIn("speaker_similarity_avg", manifest)
            self.assertEqual(manifest["segments"][0]["candidate_count"], 1)
            self.assertIsNotNone(manifest["segments"][0]["selected_seed"])
            self.assertEqual(result.reference_profile, "short")
            self.assertEqual(result.fidelity_mode, "normal")
            self.assertAlmostEqual(result.speaker_similarity_avg or 0.0, 0.83, places=3)

    def test_synthesize_spanish_maximum_fidelity_selects_best_candidate_seed(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir) / "profiles"

            def fake_similarity(_reference: str | Path, candidate: str | Path) -> float:
                candidate_name = Path(candidate).name
                if "candidate_01" in candidate_name:
                    return 0.45
                if "candidate_02" in candidate_name:
                    return 0.9
                if "candidate_03" in candidate_name:
                    return 0.6
                return 0.6

            service = SpanishVoiceService(
                backend=backend,
                jobs_dir=Path(temp_dir) / "jobs",
                speaker_profile_manager=SpeakerProfileManager(profiles_dir=profiles_dir),
                quality_analyzer=_always_valid_quality,
                speaker_similarity_fn=fake_similarity,
            )
            wav_path = Path(temp_dir) / "reference.wav"
            _create_wav(wav_path, duration_seconds=8.0)

            result = service.synthesize_spanish(
                str(wav_path),
                "Texto de prueba para ranking de candidatos.",
                fidelity_mode="maxima",
            )
            manifest = json.loads((Path(result.job_dir) / "manifest.json").read_text(encoding="utf-8"))
            selected_seed = manifest["segments"][0]["selected_seed"]

            self.assertEqual(manifest["segments"][0]["candidate_count"], 3)
            self.assertEqual(selected_seed, 42 + 1 + 1000)
            self.assertGreaterEqual(result.speaker_similarity_avg or 0.0, 0.89)

    def test_synthesize_spanish_uses_existing_speaker_profile_without_new_reference(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir) / "profiles"
            manager = SpeakerProfileManager(profiles_dir=profiles_dir)
            source_wav = Path(temp_dir) / "source.wav"
            _create_wav(source_wav, duration_seconds=12.0)
            manager.create_or_update_profile("voz_demo", source_wav, requested_reference_profile="auto")

            service = SpanishVoiceService(
                backend=backend,
                jobs_dir=Path(temp_dir) / "jobs",
                speaker_profile_manager=manager,
                quality_analyzer=_always_valid_quality,
                speaker_similarity_fn=lambda *_args, **_kwargs: 0.81,
            )

            result = service.synthesize_spanish(
                reference_audio_path=None,
                text_es="Prueba usando perfil guardado.",
                speaker_profile_name="voz_demo",
                fidelity_mode="normal",
            )

            self.assertEqual(result.speaker_profile_used, "voz_demo")
            self.assertEqual(result.segment_count, 1)

    def test_synthesize_spanish_keeps_partial_artifacts_on_failed_block(self) -> None:
        backend = FakeBackend(fail_on_call=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            profiles_dir = Path(temp_dir) / "profiles"
            service = SpanishVoiceService(
                backend=backend,
                jobs_dir=Path(temp_dir) / "jobs",
                speaker_profile_manager=SpeakerProfileManager(profiles_dir=profiles_dir),
                quality_analyzer=_always_valid_quality,
                speaker_similarity_fn=lambda *_args, **_kwargs: 0.8,
            )
            wav_path = Path(temp_dir) / "reference.wav"
            _create_wav(wav_path, duration_seconds=8.0)
            text = " ".join(["palabra"] * 220)

            with self.assertRaises(RuntimeError) as context:
                service.synthesize_spanish(str(wav_path), text, fidelity_mode="normal")

            self.assertIn("Fallo en bloque 2/", str(context.exception))
            job_dirs = list((Path(temp_dir) / "jobs").glob("job_*"))
            self.assertEqual(len(job_dirs), 1)
            job_dir = job_dirs[0]
            manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failed_segment"], 2)
            self.assertTrue((job_dir / "segments" / "segment_001.wav").exists())
            self.assertFalse((job_dir / "final.wav").exists())


if __name__ == "__main__":
    unittest.main()
