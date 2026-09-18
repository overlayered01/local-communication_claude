"""Vosk (Kaldi) Korean small model: CPU-only, streaming-capable, low accuracy. The zero-GPU floor baseline for STT."""
from __future__ import annotations

import json

import numpy as np

from llmcomm.core.interfaces import STTEngine
from llmcomm.tts._util import run_blocking
from llmcomm.core.params import Param

from ._util import ensure_model, to_16k

_URL = "https://alphacephei.com/vosk/models/{name}.zip"


class VoskSTT(STTEngine):
    streaming = True


    PARAMS = [
        Param("model", "str", "vosk-model-small-ko-0.22", description="Vosk 모델 이름", reload=True),
    ]

    def __init__(self, model: str = "vosk-model-small-ko-0.22"):
        self.model_name = self.model = model
        self._model = None

    def _load(self):
        try:
            from vosk import Model, SetLogLevel
        except ImportError as e:
            raise ImportError("pip install 'llmcomm[stt]'") from e
        SetLogLevel(-1)
        self._model = Model(str(ensure_model(self.model_name, _URL.format(name=self.model_name), "vosk")))

    def _run(self, samples: np.ndarray, sample_rate: int) -> str:
        from vosk import KaldiRecognizer

        if self._model is None:
            self._load()
        pcm = (np.clip(to_16k(samples, sample_rate), -1, 1) * 32767).astype(np.int16).tobytes()
        rec = KaldiRecognizer(self._model, 16000)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get("text", "").strip()

    async def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        return await run_blocking(self._run, samples, sample_rate)
