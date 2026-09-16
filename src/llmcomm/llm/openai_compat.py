from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from llmcomm.core.interfaces import LLMBackend
from llmcomm.core.types import Message


class OpenAICompatBackend(LLMBackend):
    """/v1/chat/completions SSE streaming. Covers vLLM, llama.cpp server, LM Studio, Ollama's compat endpoint."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str = "none", extra: dict | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.extra = extra or {}
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def stream(self, messages: list[Message], **gen_kwargs) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            **self.extra,
            **gen_kwargs,
        }
        async with self._client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                delta = json.loads(body)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    async def close(self) -> None:
        await self._client.aclose()
