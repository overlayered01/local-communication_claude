from __future__ import annotations

import numpy as np

from llmcomm.core.gpu import register_cuda_dlls
from llmcomm.core.interfaces import STTEngine
from llmcomm.tts._util import run_blocking


class FasterWhisperSTT(STTEngine):
    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cuda", compute_type: str = "float16", language: str = "ko"):
        self.model_size, self.device, self.compute_type, self.language = model_size, device, compute_type, language
        self._model = None

    def _load(self):
        register_cuda_dlls()
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError("pip install 'llmcomm[whisper]'") from e
        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def _run(self, samples: np.ndarray, sample_rate: int) -> str:
        if self._model is None:
            self._load()
        if sample_rate != 16000:
            idx = np.linspace(0, len(samples) - 1, int(len(samples) * 16000 / sample_rate))
            samples = np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)
        segments, _ = self._model.transcribe(samples, language=self.language, beam_size=1, vad_filter=False)
        return "".join(s.text for s in segments).strip()

    async def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        return await run_blocking(self._run, samples, sample_rate)
