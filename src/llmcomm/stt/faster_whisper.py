"""faster-whisper (CTranslate2) Whisper. `model` is a size name ("large-v3-turbo", "small", ...) or a
path to a converted CTranslate2 model directory, which is how Korean fine-tunes are used
(see `llmcomm stt-convert`). GPU by default; `device: cpu` + `compute_type: int8` for the CPU path."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from llmcomm.core.gpu import register_cuda_dlls
from llmcomm.core.interfaces import STTEngine
from llmcomm.tts._util import run_blocking
from llmcomm.core.params import Param

from ._util import MODELS_ROOT, to_16k


class FasterWhisperSTT(STTEngine):

    PARAMS = [
        Param("beam_size", "int", 1, 1, 8, 1, description="빔 크기. 1이 가장 빠름"),
        Param("vad_filter", "bool", False, description="무음 구간 제거(Silero VAD)"),
        Param("language", "choice", "ko", choices=["ko", "en", "ja", "zh"], description="강제 언어"),
        Param("model_size", "str", "large-v3-turbo", description="크기 이름 또는 models/ct2/<name>", reload=True),
        Param("device", "choice", "cuda", choices=["cuda", "cpu"], description="실행 장치", reload=True),
        Param("compute_type", "choice", "float16", choices=["float16", "int8_float16", "int8", "float32"], description="정밀도", reload=True),
        Param("cpu_threads", "int", 0, 0, 32, 1, description="CPU 스레드(0=자동)", reload=True),
    ]

    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cuda", compute_type: str = "float16",
                 language: str = "ko", beam_size: int = 1, vad_filter: bool = False, cpu_threads: int = 0):
        self.model_size, self.device, self.compute_type, self.language = model_size, device, compute_type, language
        self.beam_size, self.vad_filter, self.cpu_threads = beam_size, vad_filter, cpu_threads
        self._model = None

    def _load(self):
        if self.device == "cuda":
            register_cuda_dlls()
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError("pip install 'llmcomm[whisper]'") from e
        model = self.model_size
        local = MODELS_ROOT / "ct2" / model
        if local.exists():  # converted fine-tune under models/ct2/<name>
            model = str(local)
        elif Path(model).exists():
            model = str(Path(model))
        self._model = WhisperModel(model, device=self.device, compute_type=self.compute_type, cpu_threads=self.cpu_threads)

    def _run(self, samples: np.ndarray, sample_rate: int) -> str:
        if self._model is None:
            self._load()
        segments, _ = self._model.transcribe(to_16k(samples, sample_rate), language=self.language,
                                             beam_size=self.beam_size, vad_filter=self.vad_filter)
        return "".join(s.text for s in segments).strip()

    async def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        return await run_blocking(self._run, samples, sample_rate)
