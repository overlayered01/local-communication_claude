"""LLM stream -> sentence split -> TTS queue -> ordered audio chunks.

Yields (event, payload):
  ("text", delta)                text delta as it arrives
  ("sentence", str)              completed sentence handed to TTS
  ("audio", AudioChunk)          synthesized audio, in sentence order
  ("metrics", dict)              once at the end

Audio is emitted the moment its synthesis finishes (and all earlier sentences are out),
independent of the LLM stream: the LLM producer and the ordered audio drainer run as
separate tasks and merge into one event queue. Earlier versions only checked for finished
audio when the next LLM delta arrived, which added a delta-interval wait (200ms+) to first audio.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from llmcomm.core.interfaces import LLMBackend, TTSEngine
from llmcomm.core.types import AudioChunk, Message

from .sentence_splitter import StreamSentenceSplitter

_AUDIO_DONE = "_audio_done"
_ERROR = "_error"


async def converse(
    llm: LLMBackend,
    tts: TTSEngine,
    messages: list[Message],
    tts_concurrency: int = 2,
    min_chars: int = 6,
    **gen_kwargs,
) -> AsyncIterator[tuple[str, object]]:
    t0 = time.perf_counter()
    ms = lambda: (time.perf_counter() - t0) * 1000  # noqa: E731
    metrics = {"ttft_ms": None, "first_sentence_ms": None, "first_audio_ms": None, "llm_done_ms": None, "total_ms": None, "sentences": 0}

    out: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
    pending: asyncio.Queue[asyncio.Task[AudioChunk] | None] = asyncio.Queue()  # synth tasks in sentence order
    sem = asyncio.Semaphore(tts_concurrency)
    synth_tasks: list[asyncio.Task[AudioChunk]] = []

    async def synth(sentence: str) -> AudioChunk:
        async with sem:
            return await tts.synthesize(sentence)

    def submit(sentence: str) -> None:
        if metrics["first_sentence_ms"] is None:
            metrics["first_sentence_ms"] = ms()
        metrics["sentences"] += 1
        out.put_nowait(("sentence", sentence))
        task = asyncio.create_task(synth(sentence))
        synth_tasks.append(task)
        pending.put_nowait(task)

    async def produce() -> None:
        """Stream the LLM, split sentences, hand them to TTS."""
        try:
            splitter = StreamSentenceSplitter(min_chars=min_chars)
            async for delta in llm.stream(messages, **gen_kwargs):
                if metrics["ttft_ms"] is None:
                    metrics["ttft_ms"] = ms()
                out.put_nowait(("text", delta))
                for s in splitter.feed(delta):
                    submit(s)
            metrics["llm_done_ms"] = ms()
            for s in splitter.flush():
                submit(s)
        except Exception as e:  # surface in the consumer instead of dying silently
            out.put_nowait((_ERROR, e))
        finally:
            pending.put_nowait(None)

    async def drain() -> None:
        """Emit audio strictly in sentence order, each as soon as it is ready."""
        try:
            while (task := await pending.get()) is not None:
                chunk = await task
                if metrics["first_audio_ms"] is None:
                    metrics["first_audio_ms"] = ms()
                out.put_nowait(("audio", chunk))
        except Exception as e:
            out.put_nowait((_ERROR, e))
        finally:
            out.put_nowait((_AUDIO_DONE, None))

    producer = asyncio.create_task(produce())
    drainer = asyncio.create_task(drain())
    try:
        while True:
            ev, payload = await out.get()
            if ev == _AUDIO_DONE:
                break
            if ev == _ERROR:
                raise payload  # type: ignore[misc]
            yield ev, payload
        metrics["total_ms"] = ms()
        yield ("metrics", metrics)
    finally:
        for t in (producer, drainer, *synth_tasks):
            if not t.done():
                t.cancel()
        await asyncio.gather(producer, drainer, *synth_tasks, return_exceptions=True)
