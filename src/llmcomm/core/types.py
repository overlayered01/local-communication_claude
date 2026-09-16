from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class AudioChunk:
    """PCM audio. `samples` is float32 mono in [-1, 1]."""

    samples: "np.ndarray"  # noqa: F821
    sample_rate: int
    text: str = ""  # source text this chunk was synthesized from


@dataclass
class LLMMetrics:
    ttft_ms: float = 0.0  # time to first token
    total_ms: float = 0.0
    output_tokens: int = 0
    tokens_per_s: float = 0.0
    vram_peak_mb: float = 0.0
    output_text: str = ""


@dataclass
class TTSMetrics:
    first_audio_ms: float = 0.0
    total_ms: float = 0.0
    audio_sec: float = 0.0
    rtf: float = 0.0  # real-time factor = synth time / audio duration (lower is better)
    vram_peak_mb: float = 0.0
    chars: int = 0
    roundtrip_cer: float | None = None  # STT(synth(text)) vs text


@dataclass
class STTMetrics:
    total_ms: float = 0.0
    audio_sec: float = 0.0
    rtf: float = 0.0
    cer: float | None = None
    transcript: str = ""


@dataclass
class BenchRecord:
    kind: Literal["llm", "tts", "stt", "e2e"]
    option: str  # config name, e.g. "ollama_qwen3_14b"
    prompt_id: str
    metrics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
