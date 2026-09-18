"""Peak VRAM sampling via NVML. Works process-agnostic (reads whole-GPU usage)."""
from __future__ import annotations

import asyncio
import contextlib


def register_cuda_dlls() -> None:
    """Windows: make CUDA 12 / cuDNN 9 DLLs visible to non-torch libraries (onnxruntime-gpu, ctranslate2).
    The torch cu12x wheel bundles them in torch/lib; importing torch also loads them into the process."""
    import os
    import sys

    if sys.platform != "win32":
        return
    dirs: list[str] = []
    try:
        import torch  # noqa: F401

        dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:
        pass
    # pip-installed NVIDIA runtimes (nvidia-cublas-cu13 etc.) live in site-packages/nvidia/<lib>/bin
    import glob
    import site

    for sp in set(site.getsitepackages() + [site.getusersitepackages()]):
        dirs += glob.glob(os.path.join(sp, "nvidia", "*", "bin"))
        dirs += glob.glob(os.path.join(sp, "nvidia", "*", "bin", "*"))  # cu13 layout: nvidia/cu13/bin/x86_64
    for d in dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


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


def register_onnxruntime_dll() -> None:
    """Load the pip onnxruntime DLL before any extension that links `onnxruntime.dll` by name.

    Windows 11 ships an old onnxruntime.dll (1.17) in System32 for Windows ML. sherpa-onnx's extension
    module has no bundled copy, so without this it binds to the System32 DLL and crashes with
    "The requested API version [28] is not available".
    """
    import ctypes
    import os
    from pathlib import Path

    try:
        import onnxruntime
    except ImportError:
        return
    capi = Path(onnxruntime.__file__).parent / "capi"
    dll = capi / "onnxruntime.dll"
    if dll.exists():
        os.add_dll_directory(str(capi))
        ctypes.WinDLL(str(dll))
