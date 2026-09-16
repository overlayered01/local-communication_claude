from __future__ import annotations

from pathlib import Path

from llmcomm.core.interfaces import TTSEngine
from llmcomm.core.types import AudioChunk

from ._util import run_blocking, to_mono_float32


class SupertonicTTS(TTSEngine):
    """Supertone Supertonic 3 via the `supertonic` PyPI package (ONNX Runtime, 44.1kHz, 31 languages incl. Korean).

    Install: uv sync --extra supertonic
    Weights auto-download to `model_dir` on first use (HF: supertone-oss-archive/supertonic-3).
    Voices: M1-M5, F1-F5.
    """

    sample_rate = 44100

    def __init__(
        self,
        model: str = "supertonic-3",
        model_dir: str = "models/supertonic-assets",
        voice: str = "F1",
        lang: str = "ko",
        total_steps: int = 8,
        speed: float = 1.05,
        auto_download: bool = True,
        device: str = "cpu",
    ):
        self.device = device
        self.model = model
        self.model_dir = str(Path(model_dir).resolve())
        self.voice = voice
        self.lang = lang
        self.total_steps = total_steps
        self.speed = speed
        self.auto_download = auto_download
        self._tts = None
        self._styles: dict[str, object] = {}

    def _load(self):
        try:
            from supertonic import TTS
        except ImportError as e:
            raise ImportError("pip install supertonic  (or: uv sync --extra supertonic)") from e
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        if self.device == "cuda":
            # Package hardcodes CPU; the list object is shared with loader.py, so mutate in place.
            import onnxruntime as ort
            from supertonic import config as sconf

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDAExecutionProvider unavailable: uv pip install onnxruntime-gpu (replaces onnxruntime)")
            sconf.DEFAULT_ONNX_PROVIDERS[:] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._tts = TTS(model=self.model, model_dir=self.model_dir, auto_download=self.auto_download)

    def _style(self, voice: str):
        if voice not in self._styles:
            self._styles[voice] = self._tts.get_voice_style(voice_name=voice)
        return self._styles[voice]

    def _synth(self, text: str, voice: str):
        if self._tts is None:
            self._load()
        wav, _duration = self._tts.synthesize(
            text, voice_style=self._style(voice), total_steps=self.total_steps, speed=self.speed, lang=self.lang
        )
        return to_mono_float32(wav)  # wav shape (1, n) -> (n,)

    async def synthesize(self, text: str, voice: str | None = None) -> AudioChunk:
        samples = await run_blocking(self._synth, text, voice or self.voice)
        return AudioChunk(samples=samples, sample_rate=self.sample_rate, text=text)
