from __future__ import annotations

from llmcomm.core.interfaces import TTSEngine
from llmcomm.core.types import AudioChunk

from ._util import run_blocking, to_mono_float32


class MeloTTS(TTSEngine):
    """MeloTTS Korean model.

    Setup: pip install git+https://github.com/myshell-ai/MeloTTS.git ; python -m unidic download
    """

    def __init__(self, language: str = "KR", speaker: str = "KR", device: str = "cuda", speed: float = 1.0):
        self.language, self.speaker, self.device, self.speed = language, speaker, device, speed
        self._model = None

    def _load(self):
        import sys
        from pathlib import Path

        # Windows: g2pkk wants the unbuildable `eunjeon` package; use our Kiwi-backed shim instead.
        shims = str(Path(__file__).resolve().parents[3] / "shims")
        if sys.platform == "win32" and shims not in sys.path:
            sys.path.insert(0, shims)
        try:
            from melo.api import TTS
        except ImportError as e:
            raise ImportError("MeloTTS not installed. See class docstring.") from e
        self._model = TTS(language=self.language, device=self.device)
        self.sample_rate = self._model.hps.data.sampling_rate

    def _synth(self, text: str, speaker: str):
        if self._model is None:
            self._load()
        spk_id = self._model.hps.data.spk2id[speaker]
        wav = self._model.tts_to_file(text, spk_id, output_path=None, speed=self.speed, quiet=True)
        return to_mono_float32(wav)

    async def synthesize(self, text: str, voice: str | None = None) -> AudioChunk:
        samples = await run_blocking(self._synth, text, voice or self.speaker)
        return AudioChunk(samples=samples, sample_rate=self.sample_rate, text=text)
