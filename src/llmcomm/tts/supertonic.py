from __future__ import annotations

from llmcomm.core.interfaces import TTSEngine
from llmcomm.core.types import AudioChunk

from ._util import run_blocking, to_mono_float32


class SupertonicTTS(TTSEngine):
    """Supertone Supertonic (ONNX, Korean/English). Fast, CPU-capable.

    Setup: clone https://github.com/supertone-inc/supertonic and download assets per its README,
    then point `repo_dir` at the checkout. The Python API surface has changed between releases;
    adapt `_load`/`_synth` if the import fails.
    """

    def __init__(self, repo_dir: str, voice: str = "default", device: str = "cuda", **kw):
        self.repo_dir = repo_dir
        self.voice = voice
        self.device = device
        self._tts = None
        self._kw = kw

    def _load(self):
        import sys

        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        try:
            from supertonic import TTS  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"Supertonic not importable from {self.repo_dir}. Follow the repo README to install."
            ) from e
        self._tts = TTS(device=self.device, **self._kw)
        self.sample_rate = getattr(self._tts, "sample_rate", 44100)

    def _synth(self, text: str, voice: str):
        if self._tts is None:
            self._load()
        wav = self._tts.synthesize(text, voice=voice)
        return to_mono_float32(wav)

    async def synthesize(self, text: str, voice: str | None = None) -> AudioChunk:
        samples = await run_blocking(self._synth, text, voice or self.voice)
        return AudioChunk(samples=samples, sample_rate=self.sample_rate, text=text)
