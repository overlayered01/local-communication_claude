from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from llmcomm.core.interfaces import LLMBackend
from llmcomm.core.types import Message
from llmcomm.core.params import Param


class OllamaBackend(LLMBackend):
    """Ollama native /api/chat streaming."""


    PARAMS = [
        Param("temperature", "float", 0.7, 0.0, 2.0, 0.05, description="샘플링 온도. 낮으면 일관적, 높으면 다양"),
        Param("top_p", "float", 0.9, 0.0, 1.0, 0.05, description="누적 확률 컷오프"),
        Param("top_k", "int", 40, 0, 200, 1, description="상위 k 토큰만 샘플링"),
        Param("repeat_penalty", "float", 1.1, 0.8, 2.0, 0.05, description="반복 억제"),
        Param("num_ctx", "int", 8192, 1024, 32768, 1024, description="컨텍스트 길이(토큰). 크면 VRAM 증가"),
        Param("num_predict", "int", 200, 16, 2048, 16, description="최대 생성 토큰"),
        Param("seed", "int", 0, 0, 2**31 - 1, 1, description="0이면 무작위"),
        Param("think", "bool", False, description="사고(reasoning) 모드. 음성 대화에서는 꺼야 지연이 공정", reload=False),
        Param("keep_alive", "str", "10m", description="응답 후 모델을 GPU에 유지하는 시간"),
        Param("model", "str", "", description="Ollama 모델 태그", reload=True),
        Param("host", "str", "http://localhost:11434", description="Ollama 서버", reload=True),
    ]

    def __init__(self, model: str, host: str = "http://localhost:11434", options: dict | None = None, keep_alive: str = "10m", think: bool | None = None):
        self.think = think
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
        if self.think is not None:
            payload["think"] = self.think  # qwen3 etc.: disable reasoning for low-latency chat
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

    async def unload(self, wait_s: float = 20.0) -> None:
        """Evict the model from GPU (keep_alive=0) and wait until Ollama reports it gone, so the next
        benchmark's idle VRAM baseline is clean."""
        import asyncio

        try:
            await self._client.post(f"{self.host}/api/chat", json={"model": self.model, "messages": [], "keep_alive": 0}, timeout=30)
            deadline = asyncio.get_running_loop().time() + wait_s
            while asyncio.get_running_loop().time() < deadline:
                ps = (await self._client.get(f"{self.host}/api/ps", timeout=10)).json().get("models", [])
                if not any(m.get("name", "").startswith(self.model.split(":")[0]) for m in ps):
                    break
                await asyncio.sleep(0.5)
            await asyncio.sleep(1.0)  # let the driver release memory
        except Exception:
            pass

    async def close(self) -> None:
        await self.unload()
        await self._client.aclose()

    _OPTION_KEYS = {"temperature", "top_p", "top_k", "repeat_penalty", "num_ctx", "num_predict", "seed"}

    def set_param(self, name: str, value):
        """Sampling knobs live in the Ollama `options` dict; the rest are plain attributes."""
        if name in self._OPTION_KEYS:
            if name == "seed" and not value:
                self.options.pop("seed", None)
            else:
                self.options[name] = value
        else:
            setattr(self, name, value)

    def get_param(self, name: str):
        if name in self._OPTION_KEYS:
            default = next(p.default for p in self.PARAMS if p.name == name)
            return self.options.get(name, default)
        return getattr(self, name, None)


