from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from llmcomm.core.interfaces import LLMBackend
from llmcomm.core.types import Message
from llmcomm.core.params import Param


class OpenAICompatBackend(LLMBackend):
    """/v1/chat/completions SSE streaming. Covers vLLM, llama.cpp server, LM Studio, Ollama's compat endpoint."""


    PARAMS = [
        Param("temperature", "float", 0.7, 0.0, 2.0, 0.05, description="샘플링 온도"),
        Param("top_p", "float", 0.9, 0.0, 1.0, 0.05, description="누적 확률 컷오프"),
        Param("max_tokens", "int", 200, 16, 4096, 16, description="최대 생성 토큰"),
        Param("presence_penalty", "float", 0.0, -2.0, 2.0, 0.1, description="새 주제 유도"),
        Param("frequency_penalty", "float", 0.0, -2.0, 2.0, 0.1, description="반복 억제"),
        Param("model", "str", "", description="서버의 모델 이름", reload=True),
        Param("base_url", "str", "http://localhost:8000/v1", description="OpenAI 호환 엔드포인트", reload=True),
    ]

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

    _GEN_KEYS = {"temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty"}

    def set_param(self, name: str, value):
        if name in self._GEN_KEYS:
            self.extra[name] = value
        else:
            setattr(self, name, value)

    def get_param(self, name: str):
        if name in self._GEN_KEYS:
            return self.extra.get(name, next(p.default for p in self.PARAMS if p.name == name))
        return getattr(self, name, None)


