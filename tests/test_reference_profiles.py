from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from xtts_spanish_app.reference_profiles import (
    prepare_consolidated_reference,
    resolve_reference_profile,
)


def _create_pattern_audio(path: Path, total_seconds: float, sample_rate: int = 24_000) -> None:
    frame_count = int(total_seconds * sample_rate)
    signal = np.zeros((frame_count,), dtype=np.float32)
    tone_sections = [
        (int(0.10 * frame_count), int(0.20 * frame_count)),
        (int(0.45 * frame_count), int(0.60 * frame_count)),
        (int(0.75 * frame_count), int(0.90 * frame_count)),
    ]
    for start, end in tone_sections:
        for index in range(start, end):
            signal[index] = 0.2 * math.sin(2.0 * math.pi * 220.0 * (index / float(sample_rate)))
    sf.write(str(path), signal.reshape(-1, 1), sample_rate, subtype="PCM_16")


class ReferenceProfilesTests(unittest.TestCase):
    def test_resolve_reference_profile_auto_short_and_long(self) -> None:
        self.assertEqual(resolve_reference_profile("auto", 120.0), "short")
        self.assertEqual(resolve_reference_profile("auto", 240.0), "short")
        self.assertEqual(resolve_reference_profile("auto", 700.0), "long")

    def test_prepare_consolidated_reference_short_profile_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source_short.wav"
            target = Path(temp_dir) / "reference_short.wav"
            _create_pattern_audio(source, total_seconds=120.0)

            prepared = prepare_consolidated_reference(
                source_path=source,
                target_path=target,
                requested_profile="short",
            )
            output_audio, sample_rate = sf.read(str(prepared.path), always_2d=True, dtype="float32")
            output_duration = output_audio.shape[0] / float(sample_rate)
            silence_ratio = float(np.mean(np.abs(output_audio[:, 0]) < 0.02))

            self.assertEqual(prepared.profile, "short")
            self.assertTrue(target.exists())
            self.assertAlmostEqual(output_duration, 18.0, delta=1.5)
            self.assertLess(silence_ratio, 0.95)

    def test_prepare_consolidated_reference_long_profile_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source_long.wav"
            target = Path(temp_dir) / "reference_long.wav"
            _create_pattern_audio(source, total_seconds=720.0)

            prepared = prepare_consolidated_reference(
                source_path=source,
                target_path=target,
                requested_profile="long",
            )
            output_audio, sample_rate = sf.read(str(prepared.path), always_2d=True, dtype="float32")
            output_duration = output_audio.shape[0] / float(sample_rate)

            self.assertEqual(prepared.profile, "long")
            self.assertTrue(target.exists())
            self.assertAlmostEqual(output_duration, 24.0, delta=1.5)
            self.assertEqual(len(prepared.windows), 3)


if __name__ == "__main__":
    unittest.main()
