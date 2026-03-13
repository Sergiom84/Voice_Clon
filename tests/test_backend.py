from __future__ import annotations

import unittest

import numpy as np

from xtts_spanish_app.backend import (
    XTTSVoiceBackend,
    _is_supported_cuda_runtime,
    _patch_torch_load_compat,
    _patch_torchaudio_load_compat,
)


class _FakeCuda:
    def __init__(self, available: bool, capability: tuple[int, int], arch_list: list[str]) -> None:
        self._available = available
        self._capability = capability
        self._arch_list = arch_list

    def is_available(self) -> bool:
        return self._available

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return self._capability

    def get_arch_list(self) -> list[str]:
        return self._arch_list

    def get_device_name(self, index: int) -> str:
        return "Fake GPU"


class _FakeTorch:
    def __init__(self, cuda: _FakeCuda | None = None) -> None:
        self.cuda = cuda or _FakeCuda(False, (0, 0), [])
        self.calls: list[dict[str, object]] = []

    def load(self, *args, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class _FakeTTSMissingTorchCodec:
    def tts(self, *args, **kwargs):
        raise ImportError("TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.")


class _FakeTTSBrokenTorchCodec:
    def tts(self, *args, **kwargs):
        raise RuntimeError("Could not load libtorchcodec. Native dependency load failed.")


class _FakeTorchArrayModule:
    float32 = np.float32

    def tensor(self, data, dtype=None):
        return np.array(data, dtype=dtype)


class _FakeSoundFile:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def read(self, uri, always_2d=True, dtype="float32"):
        self.calls.append(uri)
        return np.array([[0.1], [0.2], [0.3]], dtype=np.float32), 24_000


class _FakeTorchAudio:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def load(self, *args, **kwargs):
        raise self._error


class BackendCompatibilityTests(unittest.TestCase):
    def test_patch_torch_load_sets_weights_only_false_by_default(self) -> None:
        fake_torch = _FakeTorch()

        _patch_torch_load_compat(fake_torch)
        fake_torch.load("checkpoint.pt")
        fake_torch.load("checkpoint.pt", weights_only=True)

        self.assertEqual(fake_torch.calls[0]["weights_only"], False)
        self.assertEqual(fake_torch.calls[1]["weights_only"], True)

    def test_supported_cuda_runtime_returns_true_for_matching_arch(self) -> None:
        fake_torch = _FakeTorch(cuda=_FakeCuda(True, (8, 6), ["sm_86", "sm_90"]))

        supported, warning = _is_supported_cuda_runtime(fake_torch)

        self.assertTrue(supported)
        self.assertIsNone(warning)

    def test_supported_cuda_runtime_returns_warning_for_unsupported_arch(self) -> None:
        fake_torch = _FakeTorch(cuda=_FakeCuda(True, (12, 0), ["sm_86", "sm_90"]))

        supported, warning = _is_supported_cuda_runtime(fake_torch)

        self.assertFalse(supported)
        self.assertIn("sm_120", warning)
        self.assertIn("CU128", warning)

    def test_synthesize_fragment_surfaces_torchcodec_install_hint(self) -> None:
        backend = XTTSVoiceBackend()
        backend._tts = _FakeTTSMissingTorchCodec()

        with self.assertRaises(RuntimeError) as context:
            backend.synthesize_fragment("Hola", "referencia.wav")

        self.assertIn("python -m pip install torchcodec", str(context.exception))

    def test_synthesize_fragment_surfaces_torchcodec_native_hint(self) -> None:
        backend = XTTSVoiceBackend()
        backend._tts = _FakeTTSBrokenTorchCodec()

        with self.assertRaises(RuntimeError) as context:
            backend.synthesize_fragment("Hola", "referencia.wav")

        self.assertIn("FFmpeg 4-7", str(context.exception))

    def test_patch_torchaudio_load_uses_soundfile_fallback_for_torchcodec_errors(self) -> None:
        fake_torchaudio = _FakeTorchAudio(
            ImportError("TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.")
        )
        fake_soundfile = _FakeSoundFile()
        fake_torch = _FakeTorchArrayModule()

        _patch_torchaudio_load_compat(
            torchaudio_module=fake_torchaudio,
            soundfile_module=fake_soundfile,
            torch_module=fake_torch,
        )
        audio, sample_rate = fake_torchaudio.load("referencia.wav")

        self.assertEqual(sample_rate, 24_000)
        self.assertEqual(audio.shape, (1, 3))
        self.assertEqual(fake_soundfile.calls, ["referencia.wav"])


if __name__ == "__main__":
    unittest.main()
