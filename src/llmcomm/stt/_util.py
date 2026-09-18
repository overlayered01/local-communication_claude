from __future__ import annotations

import io
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from llmcomm.core.registry import CONFIG_ROOT

MODELS_ROOT = CONFIG_ROOT.parent / "models"


def to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono float32 at 16 kHz, which every STT engine here expects."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sample_rate == 16000:
        return x
    n = int(len(x) * 16000 / sample_rate)
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def ensure_model(name: str, url: str, subdir: str) -> Path:
    """Download + extract a model archive into models/<subdir>/<name> on first use."""
    target = MODELS_ROOT / subdir / name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[llmcomm] downloading {url}")
    data = urllib.request.urlopen(url, timeout=120).read()
    if url.endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(target.parent)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tf:
            tf.extractall(target.parent)
    if not target.exists():
        raise FileNotFoundError(f"archive did not produce {target}")
    return target


def pick(dir_: Path, pattern: str, int8: bool) -> str:
    """Choose one file matching a glob, preferring the int8 or float variant as requested."""
    cands = sorted(dir_.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"no {pattern} in {dir_}")
    want = [c for c in cands if ("int8" in c.name) == int8]
    return str((want or cands)[0])
