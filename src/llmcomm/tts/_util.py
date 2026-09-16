from __future__ import annotations

import asyncio
import io
from typing import Callable, TypeVar

import numpy as np
import soundfile as sf

T = TypeVar("T")


async def run_blocking(fn: Callable[..., T], *args, **kw) -> T:
    return await asyncio.get_running_loop().run_in_executor(None, lambda: fn(*args, **kw))


def decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode wav/mp3/ogg bytes to float32 mono."""
    samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return samples.mean(axis=1), sr


def to_mono_float32(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr.mean(axis=0 if arr.shape[0] < arr.shape[1] else 1)
    if arr.dtype != np.float32:
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        else:
            arr = arr.astype(np.float32)
    return arr
