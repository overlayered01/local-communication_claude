from __future__ import annotations

import asyncio
from typing import AsyncIterator

from llmcomm.core.interfaces import LLMBackend
from llmcomm.core.types import Message


class LlamaCppBackend(LLMBackend):
    """In-process llama-cpp-python. Useful to measure raw engine cost without a server hop."""

    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1, chat_format: str | None = None, **kw):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError("pip install 'llmcomm[llamacpp]'") from e
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, chat_format=chat_format, verbose=False, **kw)

    async def stream(self, messages: list[Message], **gen_kwargs) -> AsyncIterator[str]:
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
