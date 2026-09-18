"""The pipeline must emit audio as soon as TTS finishes, not when the next LLM delta arrives."""
import asyncio
import time

import numpy as np

from llmcomm.core.interfaces import LLMBackend, TTSEngine
from llmcomm.core.types import AudioChunk, Message
from llmcomm.pipeline.streaming import converse


class SlowLLM(LLMBackend):
    """Two sentences; a long gap between the first sentence's end and the next delta."""

    def __init__(self, gap: float):
        self.gap = gap

    async def stream(self, messages, **gen):
        yield "첫 번째 문장입니다. "
        await asyncio.sleep(self.gap)
        yield "두 번째 문장입니다."


class FastTTS(TTSEngine):
    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self.calls: list[str] = []

    async def synthesize(self, text, voice=None):
        self.calls.append(text)
        await asyncio.sleep(self.delay)
        return AudioChunk(np.zeros(240, dtype=np.float32), 24000, text=text)


class FailingLLM(LLMBackend):
    async def stream(self, messages, **gen):
        yield "안녕하세요. "
        raise RuntimeError("boom")


async def _run(llm, tts):
    events, stamps = [], []
    t0 = time.perf_counter()
    async for ev, p in converse(llm, tts, [Message("user", "x")]):
        events.append((ev, p))
        stamps.append((ev, time.perf_counter() - t0))
    return events, stamps


async def test_audio_emitted_before_next_llm_delta():
    gap = 0.4
    events, stamps = await _run(SlowLLM(gap), FastTTS(0.02))
    first_audio = next(t for ev, t in stamps if ev == "audio")
    second_text = [t for ev, t in stamps if ev == "text"][1]
    assert first_audio < second_text, "first audio must not wait for the next LLM delta"
    assert first_audio < gap / 2
    m = dict(events)["metrics"]
    assert m["first_audio_ms"] < gap * 1000 / 2
    assert m["sentences"] == 2


async def test_audio_order_and_event_shape():
    events, _ = await _run(SlowLLM(0.01), FastTTS(0.01))
    kinds = [ev for ev, _ in events]
    assert kinds.count("sentence") == 2 and kinds.count("audio") == 2
    assert kinds[-1] == "metrics"
    audio_texts = [p.text for ev, p in events if ev == "audio"]
    assert audio_texts == ["첫 번째 문장입니다.", "두 번째 문장입니다."]
    # each sentence event precedes its own audio event
    assert kinds.index("sentence") < kinds.index("audio")


async def test_llm_error_propagates_and_cleans_up():
    tts = FastTTS(0.01)
    try:
        await _run(FailingLLM(), tts)
    except RuntimeError as e:
        assert str(e) == "boom"
    else:
        raise AssertionError("error not propagated")
    await asyncio.sleep(0.05)
    assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
