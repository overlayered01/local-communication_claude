"""Peak VRAM sampling via NVML. Works process-agnostic (reads whole-GPU usage)."""
from __future__ import annotations

import asyncio
import contextlib


def read_used_mb(device_index: int = 0) -> float:
    """One-shot whole-GPU used memory in MB (0 if NVML unavailable)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024 / 1024
    except Exception:
        return 0.0


class VRAMMonitor:
    def __init__(self, device_index: int = 0, interval: float = 0.05):
        self.device_index = device_index
        self.interval = interval
        self.baseline_mb = 0.0
        self.peak_mb = 0.0
        self._task: asyncio.Task | None = None
        self._handle = None

    def _read_mb(self) -> float:
        import pynvml

        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.used / 1024 / 1024

    async def __aenter__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.baseline_mb = self.peak_mb = self._read_mb()
            self._task = asyncio.create_task(self._poll())
        except Exception:
            self._handle = None
        return self

    async def _poll(self):
        while True:
            with contextlib.suppress(Exception):
                self.peak_mb = max(self.peak_mb, self._read_mb())
            await asyncio.sleep(self.interval)

    async def __aexit__(self, *exc):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    @property
    def delta_mb(self) -> float:
        return max(0.0, self.peak_mb - self.baseline_mb)
