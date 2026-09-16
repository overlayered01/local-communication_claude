"""Gapless local playback of ordered AudioChunks with interrupt support."""
from __future__ import annotations

import asyncio

import numpy as np

from llmcomm.core.types import AudioChunk


class AudioPlayer:
    def __init__(self):
        self._q: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.create_task(self._run())

    async def put(self, chunk: AudioChunk):
        await self._q.put(chunk)

    async def finish(self):
        await self._q.put(None)
        if self._task:
            await self._task

    def interrupt(self):
        self._stop.set()

    async def _run(self):
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        while True:
            chunk = await self._q.get()
            if chunk is None or self._stop.is_set():
                break
            data = np.ascontiguousarray(chunk.samples, dtype=np.float32)
            await loop.run_in_executor(None, lambda: (sd.play(data, chunk.sample_rate), sd.wait()))
        sd.stop()
