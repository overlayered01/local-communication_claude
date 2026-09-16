from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from llmcomm.core.interfaces import LLMBackend
from llmcomm.core.types import Message


class OllamaBackend(LLMBackend):
    """Ollama native /api/chat streaming."""

    def __init__(self, model: str, host: str = "http://localhost:11434", options: dict | None = None, keep_alive: str = "10m"):
        self.model = model
        self.host = host.rstrip("/")
        self.options = options or {}
        self.keep_alive = keep_alive
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    async def stream(self, messages: list[Message], **gen_kwargs) -> AsyncIterator[str]:
        options = dict(self.options)
        if "max_tokens" in gen_kwargs:
            options["num_predict"] = gen_kwargs.pop("max_tokens")
        options.update(gen_kwargs)
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        async with self._client.stream("POST", f"{self.host}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                delta = data.get("message", {}).get("content", "")
                if delta:
                    yield delta
                if data.get("done"):
                    break

    async def close(self) -> None:
        await self._client.aclose()
