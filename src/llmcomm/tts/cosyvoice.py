from __future__ import annotations

from llmcomm.core.interfaces import TTSEngine
from llmcomm.core.types import AudioChunk
from llmcomm.core.params import Param

from ._util import run_blocking, to_mono_float32


class CosyVoiceTTS(TTSEngine):
    """CosyVoice2 zero-shot voice cloning. Heavy (2-4GB VRAM). Requires the CosyVoice repo on sys.path.

    Setup: git clone --recursive https://github.com/FunAudioLLM/CosyVoice ; download CosyVoice2-0.5B.
    """


    PARAMS = [
        Param("prompt_wav", "str", "", description="클로닝 참조 음성", reload=True),
        Param("prompt_text", "str", "", description="참조 음성의 전사", reload=True),
        Param("fp16", "bool", True, description="반정밀도", reload=True),
    ]

    def __init__(self, repo_dir: str, model_dir: str, prompt_wav: str, prompt_text: str, fp16: bool = True):
        self.repo_dir = repo_dir
        self.model_dir = model_dir
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.fp16 = fp16
        self._model = None

    def _load(self):
        import sys

        for p in (self.repo_dir, f"{self.repo_dir}/third_party/Matcha-TTS"):
            if p not in sys.path:
                sys.path.insert(0, p)
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
        from cosyvoice.utils.file_utils import load_wav  # type: ignore

        self._model = CosyVoice2(self.model_dir, load_jit=False, load_trt=False, fp16=self.fp16)
        self._prompt = load_wav(self.prompt_wav, 16000)
        self.sample_rate = self._model.sample_rate

    def _synth(self, text: str):
        if self._model is None:
            self._load()
        import torch

        outs = [o["tts_speech"] for o in self._model.inference_zero_shot(text, self.prompt_text, self._prompt, stream=False)]
        return to_mono_float32(torch.cat(outs, dim=1).squeeze(0).cpu().numpy())

    async def synthesize(self, text: str, voice: str | None = None) -> AudioChunk:
        samples = await run_blocking(self._synth, text)
        return AudioChunk(samples=samples, sample_rate=self.sample_rate, text=text)
