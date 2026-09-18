from __future__ import annotations

import asyncio
from typing import AsyncIterator

from llmcomm.core.interfaces import LLMBackend
from llmcomm.core.types import Message
from llmcomm.core.params import Param


class LlamaCppBackend(LLMBackend):
    """In-process llama-cpp-python. Useful to measure raw engine cost without a server hop."""


    PARAMS = [
        Param("temperature", "float", 0.7, 0.0, 2.0, 0.05, description="샘플링 온도"),
        Param("top_p", "float", 0.9, 0.0, 1.0, 0.05, description="누적 확률 컷오프"),
        Param("max_tokens", "int", 200, 16, 4096, 16, description="최대 생성 토큰"),
        Param("n_ctx", "int", 8192, 1024, 32768, 1024, description="컨텍스트 길이", reload=True),
        Param("n_gpu_layers", "int", -1, -1, 200, 1, description="-1이면 전부 GPU", reload=True),
        Param("model_path", "str", "", description="GGUF 경로", reload=True),
    ]

    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1, chat_format: str | None = None, **kw):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError("pip install 'llmcomm[llamacpp]'") from e
        self.model_path, self.n_ctx, self.n_gpu_layers = model_path, n_ctx, n_gpu_layers
        self.gen_defaults: dict = {}
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, chat_format=chat_format, verbose=False, **kw)

    async def stream(self, messages: list[Message], **gen_kwargs) -> AsyncIterator[str]:
        gen_kwargs = {**self.gen_defaults, **gen_kwargs}
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[str | None] = asyncio.Queue()

        def run():
            for chunk in self._llm.create_chat_completion(
                messages=[{"role": m.role, "content": m.content} for m in messages], stream=True, **gen_kwargs
            ):
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    loop.call_soon_threadsafe(q.put_nowait, delta)
            loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, run)
        while (item := await q.get()) is not None:
            yield item
        await fut

    _GEN_KEYS = {"temperature", "top_p", "max_tokens"}

    def set_param(self, name: str, value):
        if name in self._GEN_KEYS:
            self.gen_defaults[name] = value
        else:
            setattr(self, name, value)

    def get_param(self, name: str):
        if name in self._GEN_KEYS:
            return self.gen_defaults.get(name, next(p.default for p in self.PARAMS if p.name == name))
        return getattr(self, name, None)


