"""LLM stream -> sentence split -> TTS queue -> ordered audio chunks.

Yields (event, payload):
  ("text", delta)                text delta as it arrives
  ("sentence", str)              completed sentence handed to TTS
  ("audio", AudioChunk)          synthesized audio, in sentence order
  ("metrics", dict)              once at the end
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from llmcomm.core.interfaces import LLMBackend, TTSEngine
from llmcomm.core.types import AudioChunk, Message

from .sentence_splitter import StreamSentenceSplitter


async def converse(
    llm: LLMBackend,
    tts: TTSEngine,
    messages: list[Message],
    tts_concurrency: int = 2,
    **gen_kwargs,
) -> AsyncIterator[tuple[str, object]]:
    t0 = time.perf_counter()
    splitter = StreamSentenceSplitter()
    sem = asyncio.Semaphore(tts_concurrency)
    tasks: list[asyncio.Task[AudioChunk]] = []
    metrics = {"ttft_ms": None, "first_sentence_ms": None, "first_audio_ms": None, "llm_done_ms": None, "total_ms": None, "sentences": 0}

    async def synth(sentence: str) -> AudioChunk:
        async with sem:
            return await tts.synthesize(sentence)

    def submit(sentence: str):
        if metrics["first_sentence_ms"] is None:
            metrics["first_sentence_ms"] = (time.perf_counter() - t0) * 1000
        metrics["sentences"] += 1
        tasks.append(asyncio.create_task(synth(sentence)))

    next_idx = 0

    async def drain_ready():
        """Yield audio strictly in order, but only for tasks already finished (non-blocking)."""
        nonlocal next_idx
        while next_idx < len(tasks) and tasks[next_idx].done():
            chunk = tasks[next_idx].result()
            next_idx += 1
            if metrics["first_audio_ms"] is None:
                metrics["first_audio_ms"] = (time.perf_counter() - t0) * 1000
            yield chunk

    async for delta in llm.stream(messages, **gen_kwargs):
        if metrics["ttft_ms"] is None:
            metrics["ttft_ms"] = (time.perf_counter() - t0) * 1000
        yield ("text", delta)
        for s in splitter.feed(delta):
            yield ("sentence", s)
            submit(s)
        async for chunk in drain_ready():
            yield ("audio", chunk)

    metrics["llm_done_ms"] = (time.perf_counter() - t0) * 1000
    for s in splitter.flush():
        yield ("sentence", s)
        submit(s)

    while next_idx < len(tasks):
        chunk = await tasks[next_idx]
        next_idx += 1
        if metrics["first_audio_ms"] is None:
            metrics["first_audio_ms"] = (time.perf_counter() - t0) * 1000
        yield ("audio", chunk)

    metrics["total_ms"] = (time.perf_counter() - t0) * 1000
    yield ("metrics", metrics)
