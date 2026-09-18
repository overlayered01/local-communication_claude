"""sherpa-onnx STT: CPU/ONNX engines that do not compete with the LLM for the GPU.

Three model families are wired up, all downloaded on first use from the k2-fsa release page:
  sense_voice          FunAudioLLM SenseVoice-Small (zh/en/ja/ko/yue), offline, very fast on CPU
  zipformer            Korean zipformer transducer, offline
  zipformer_streaming  Korean streaming zipformer; here the whole clip is fed as one stream,
                       but the engine supports true incremental decoding (partial results).
"""
from __future__ import annotations

import numpy as np

from llmcomm.core.gpu import register_onnxruntime_dll
from llmcomm.core.interfaces import STTEngine
from llmcomm.tts._util import run_blocking
from llmcomm.core.params import Param

from ._util import ensure_model, pick, to_16k

_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"


class SherpaOnnxSTT(STTEngine):
    streaming = False


    PARAMS = [
        Param("num_threads", "int", 4, 1, 16, 1, description="ONNX 스레드 수", reload=True),
        Param("int8", "bool", True, description="int8 양자화 모델 사용", reload=True),
        Param("language", "choice", "ko", choices=["ko", "auto", "zh", "en", "ja", "yue"], description="SenseVoice 언어 태그", reload=True),
        Param("model", "str", "", description="k2-fsa 릴리스 모델 디렉터리 이름", reload=True),
    ]

    def __init__(self, family: str, model: str, int8: bool = True, num_threads: int = 4, language: str = "ko", provider: str = "cpu"):
        self.family, self.model_name, self.int8, self.num_threads, self.language, self.provider = family, model, int8, num_threads, language, provider
        self.model = model
        self.streaming = family == "zipformer_streaming"
        self._rec = None

    def _load(self):
        register_onnxruntime_dll()  # must precede the sherpa import (System32 DLL collision)
        try:
            import sherpa_onnx
        except ImportError as e:
            raise ImportError("pip install 'llmcomm[stt]'") from e
        d = ensure_model(self.model_name, f"{_RELEASE}{self.model_name}.tar.bz2", "sherpa-onnx")
        common = dict(tokens=str(d / "tokens.txt"), num_threads=self.num_threads, provider=self.provider, debug=False)
        if self.family == "sense_voice":
            self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=pick(d, "model*.onnx", self.int8), language=self.language, use_itn=True, **common)
        elif self.family == "zipformer":
            self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=pick(d, "encoder*.onnx", self.int8), decoder=pick(d, "decoder*.onnx", False),
                joiner=pick(d, "joiner*.onnx", self.int8), decoding_method="greedy_search", **common)
        elif self.family == "zipformer_streaming":
            self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=pick(d, "encoder*.onnx", self.int8), decoder=pick(d, "decoder*.onnx", False),
                joiner=pick(d, "joiner*.onnx", self.int8), decoding_method="greedy_search",
                enable_endpoint_detection=False, **common)
        else:
            raise ValueError(f"unknown sherpa family {self.family}")

    def _run(self, samples: np.ndarray, sample_rate: int) -> str:
        if self._rec is None:
            self._load()
        x = to_16k(samples, sample_rate)
        s = self._rec.create_stream()
        s.accept_waveform(16000, x)
        if self.streaming:
            s.accept_waveform(16000, np.zeros(int(0.8 * 16000), dtype=np.float32))  # tail padding flushes the encoder
            s.input_finished()
            while self._rec.is_ready(s):
                self._rec.decode_stream(s)
            return self._rec.get_result(s).strip()
        self._rec.decode_stream(s)
        return s.result.text.strip()

    async def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        return await run_blocking(self._run, samples, sample_rate)
