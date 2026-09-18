from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

import numpy as np

from .types import AudioChunk, Message


class LLMBackend(ABC):
    """Streaming chat backend. Every option under test implements this."""

    name: str = "llm"

    @abstractmethod
    async def stream(self, messages: list[Message], **gen_kwargs) -> AsyncIterator[str]:
        """Yield text deltas as they arrive."""
        ...

    async def complete(self, messages: list[Message], **gen_kwargs) -> str:
        return "".join([d async for d in self.stream(messages, **gen_kwargs)])

    async def warmup(self) -> None:
        await self.complete([Message("user", "안녕")], max_tokens=4)

    async def close(self) -> None:
        pass


class TTSEngine(ABC):
    name: str = "tts"
    sample_rate: int = 24000

    @abstractmethod
    async def synthesize(self, text: str, voice: str | None = None) -> AudioChunk:
        """Synthesize one sentence-sized text into audio."""
        ...

    async def stream(self, text: str, voice: str | None = None) -> AsyncIterator[AudioChunk]:
        """Engines supporting chunked output override this; default is one chunk."""
        yield await self.synthesize(text, voice)

    async def warmup(self) -> None:
        await self.synthesize("안녕하세요.")

    async def close(self) -> None:
        pass


class STTEngine(ABC):
    name: str = "stt"
    streaming: bool = False  # True if the engine can decode incrementally (partial results while the user speaks)

    @abstractmethod
    async def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        ...

    async def warmup(self) -> None:
        await self.transcribe(np.zeros(16000, dtype=np.float32), 16000)

    async def close(self) -> None:
        pass
