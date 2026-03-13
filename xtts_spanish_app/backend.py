from __future__ import annotations

from abc import ABC, abstractmethod
import os

from .settings import DEFAULT_OUTPUT_SAMPLE_RATE, DEFAULT_XTTS_MODEL, DEFAULT_SYNTHESIS_SEED


class VoiceBackend(ABC):
    sample_rate = DEFAULT_OUTPUT_SAMPLE_RATE

    @abstractmethod
    def load_model(self) -> None:
        """Carga el modelo una sola vez."""

    @abstractmethod
    def synthesize_fragment(
        self,
        text: str,
        reference_audio_paths: list[str],
        seed: int | None = None,
    ) -> list[float]:
        """Genera un fragmento de audio mono."""

    @abstractmethod
    def describe_runtime(self) -> str:
        """Devuelve un texto corto con el estado del backend."""


class XTTSVoiceBackend(VoiceBackend):
    def __init__(self, model_name: str = DEFAULT_XTTS_MODEL, seed: int = DEFAULT_SYNTHESIS_SEED) -> None:
        self.model_name = model_name
        self._tts = None
        self._device = "cpu"
        self._last_warning = None
        self._seed = seed

    def load_model(self) -> None:
        if self._tts is not None:
            return

        try:
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            import torch
            _patch_torch_load_compat(torch)
            _patch_torchaudio_load_compat(torch_module=torch)
            from TTS.api import TTS
        except EOFError as exc:  # pragma: no cover - interactivo/licencia
            raise RuntimeError(
                "XTTS requiere aceptar la licencia del modelo Coqui antes de descargarlo. "
                "Vuelve a lanzar la app con COQUI_TOS_AGREED=1 o prepara el modelo una vez de forma interactiva."
            ) from exc
        except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError(
                "Faltan dependencias de XTTS. Ejecuta .\\install.ps1 para instalar PyTorch, Gradio y TTS."
            ) from exc

        preferred_device = "cpu"
        gpu_warning = None
        gpu_supported, gpu_warning = _is_supported_cuda_runtime(torch)
        if gpu_supported:
            preferred_device = "cuda"

        try:
            self._tts = TTS(self.model_name).to(preferred_device)
            self._device = preferred_device
            self._last_warning = gpu_warning
        except Exception as exc:  # pragma: no cover - depende de red/modelo/entorno
            if isinstance(exc, EOFError):
                raise RuntimeError(
                    "XTTS requiere aceptar la licencia del modelo Coqui antes de descargarlo. "
                    "Vuelve a lanzar la app con COQUI_TOS_AGREED=1 o prepara el modelo una vez de forma interactiva."
                ) from exc
            if preferred_device == "cuda":
                try:
                    self._tts = TTS(self.model_name).to("cpu")
                    self._device = "cpu"
                    self._last_warning = f"XTTS se ha cargado en CPU tras fallar en GPU: {exc}"
                except Exception as cpu_exc:
                    if isinstance(cpu_exc, EOFError):
                        raise RuntimeError(
                            "XTTS requiere aceptar la licencia del modelo Coqui antes de descargarlo. "
                            "Vuelve a lanzar la app con COQUI_TOS_AGREED=1 o prepara el modelo una vez de forma interactiva."
                        ) from cpu_exc
                    raise RuntimeError(
                        f"No se pudo cargar XTTS-v2 ({self.model_name}) ni en GPU ni en CPU. Error GPU: {exc}. Error CPU: {cpu_exc}"
                    ) from cpu_exc
            else:
                raise RuntimeError(
                    f"No se pudo cargar XTTS-v2 ({self.model_name}) en CPU. Error: {exc}"
                ) from exc

    def synthesize_fragment(
        self,
        text: str,
        reference_audio_paths: list[str],
        seed: int | None = None,
    ) -> list[float]:
        self.load_model()
        effective_seed = seed if seed is not None else self._seed
        self._set_seed(effective_seed)
        try:
            audio = self._tts.tts(
                text=text,
                speaker_wav=reference_audio_paths,
                language="es",
            )
        except ImportError as exc:
            message = str(exc)
            if "TorchCodec is required for load_with_torchcodec" in message or "No module named 'torchcodec'" in message:
                raise RuntimeError(
                    "Falta la dependencia 'torchcodec' en el entorno actual. "
                    "Instalala con 'python -m pip install torchcodec' y reinicia la app."
                ) from exc
            raise
        except RuntimeError as exc:
            message = str(exc)
            if "Could not load libtorchcodec" in message:
                raise RuntimeError(
                    "TorchCodec esta instalado, pero no puede cargar sus librerias nativas. "
                    "En Windows, instala FFmpeg 4-7 dentro del entorno y ejecuta la app con el entorno Conda activado. "
                    "Si mantienes un stack PyTorch CUDA, usa un build de TorchCodec compatible; la documentacion oficial "
                    "recomienda conda-forge para Windows con GPU."
                ) from exc
            raise
        if hasattr(audio, "tolist"):
            audio = audio.tolist()
        return [float(sample) for sample in audio]

    def _set_seed(self, seed: int) -> None:
        """Configura la semilla para reproducibilidad en torch y random."""
        import random

        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def describe_runtime(self) -> str:
        status = "modelo cargado" if self._tts is not None else "modelo pendiente de carga"
        if self._last_warning:
            return f"XTTS-v2 en {self._device} ({status}). Aviso: {self._last_warning}"
        return f"XTTS-v2 en {self._device} ({status})"


def _patch_torch_load_compat(torch_module) -> None:
    original_load = getattr(torch_module, "load", None)
    if original_load is None or getattr(original_load, "__xtts_spanish_app_patched__", False):
        return

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    patched_load.__xtts_spanish_app_patched__ = True
    torch_module.load = patched_load


def _patch_torchaudio_load_compat(torchaudio_module=None, soundfile_module=None, torch_module=None) -> None:
    try:
        if torchaudio_module is None:
            import torchaudio as torchaudio_module
        if soundfile_module is None:
            import soundfile as soundfile_module
        if torch_module is None:
            import torch as torch_module
    except ModuleNotFoundError:
        return

    original_load = getattr(torchaudio_module, "load", None)
    if original_load is None or getattr(original_load, "__xtts_spanish_app_patched__", False):
        return

    def patched_load(uri, *args, **kwargs):
        try:
            return original_load(uri, *args, **kwargs)
        except (ImportError, RuntimeError) as exc:
            message = str(exc)
            if "TorchCodec is required for load_with_torchcodec" not in message and "Could not load libtorchcodec" not in message:
                raise

            try:
                audio_data, sample_rate = soundfile_module.read(uri, always_2d=True, dtype="float32")
            except Exception:
                raise exc

            tensor = torch_module.tensor(audio_data.T.tolist(), dtype=torch_module.float32)
            frame_offset = kwargs.get("frame_offset", args[0] if len(args) > 0 else 0)
            num_frames = kwargs.get("num_frames", args[1] if len(args) > 1 else -1)
            channels_first = kwargs.get("channels_first", args[3] if len(args) > 3 else True)

            if frame_offset:
                tensor = tensor[:, int(frame_offset) :]
            if num_frames not in (-1, None):
                tensor = tensor[:, : int(num_frames)]
            if not channels_first:
                tensor = tensor.transpose(0, 1)

            return tensor, int(sample_rate)

    patched_load.__xtts_spanish_app_patched__ = True
    torchaudio_module.load = patched_load


def _is_supported_cuda_runtime(torch_module) -> tuple[bool, str | None]:
    if not torch_module.cuda.is_available():
        return False, None

    try:
        capability = torch_module.cuda.get_device_capability(0)
        current_arch = f"sm_{capability[0]}{capability[1]}"
        supported_archs = set(torch_module.cuda.get_arch_list())
        if current_arch in supported_archs:
            return True, None

        device_name = torch_module.cuda.get_device_name(0)
        supported_text = ", ".join(sorted(supported_archs))
        return (
            False,
            "GPU detectada "
            f"({device_name}, {current_arch}), pero el PyTorch instalado solo soporta {supported_text}. "
            "La app usara CPU. Reinstala con .\\install.ps1 -Device CU128 para usar la RTX 50.",
        )
    except Exception as exc:  # pragma: no cover - depende de CUDA
        return False, f"No se pudo verificar la compatibilidad CUDA. La app usara CPU. Detalle: {exc}"
