"""Benchmark runners. Each returns a BenchRecord list and appends to reports/results.jsonl."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import soundfile as sf

from llmcomm.core.gpu import VRAMMonitor, read_used_mb
from llmcomm.core.registry import build
from llmcomm.core.types import BenchRecord, Message
from llmcomm.pipeline.streaming import converse

from .metrics import approx_tokens, cer

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"
PROMPTS = ROOT / "data" / "prompts"
DEFAULT_SYSTEM = "당신은 친근한 한국어 대화 상대입니다. 짧고 자연스럽게 답하세요."


def load_prompts(name: str) -> list[dict]:
    path = PROMPTS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_messages(p: dict, system: str | None = None) -> list[Message]:
    """Prompt rows are either {"user": ...} or multi-turn {"messages": [...]}."""
    msgs = [Message("system", system or p.get("system", DEFAULT_SYSTEM))]
    if "messages" in p:
        msgs += [Message(m["role"], m["content"]) for m in p["messages"]]
    else:
        msgs.append(Message("user", p["user"]))
    return msgs


def _append(records: list[BenchRecord], out: Path | None = None) -> None:
    out = out or REPORTS / "results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


async def bench_llm(option: str, prompt_set: str = "korean_chat", repeats: int = 1, system: str | None = None, **gen) -> list[BenchRecord]:
    idle_mb = read_used_mb()
    llm = build("llm", option)
    prompts = load_prompts(prompt_set)
    records: list[BenchRecord] = []
    try:
        await llm.warmup()
        for p in prompts:
            for rep in range(repeats):
                msgs = _build_messages(p, system)
                t0 = time.perf_counter()
                ttft = None
                out: list[str] = []
                async with VRAMMonitor() as vram:
                    async for d in llm.stream(msgs, **gen):
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        out.append(d)
                total = (time.perf_counter() - t0) * 1000
                text = "".join(out)
                toks = approx_tokens(text)
                gen_ms = max(1.0, total - (ttft or 0))
                records.append(BenchRecord("llm", option, p["id"], {
                    "ttft_ms": round(ttft or 0, 1),
                    "total_ms": round(total, 1),
                    "output_tokens_est": toks,
                    "tokens_per_s_est": round(toks / (gen_ms / 1000), 1),
                    "vram_used_mb": round(vram.peak_mb),
                    "vram_delta_mb": round(vram.peak_mb - idle_mb),
                    "output_text": text,
                }, {"rep": rep}))
    finally:
        await llm.close()
    _append(records)
    return records


async def bench_tts(option: str, prompt_set: str = "korean_tts", stt_option: str | None = None, save_audio: bool = True) -> list[BenchRecord]:
    stt = build("stt", stt_option) if stt_option else None
    if stt:  # load the STT model first so its VRAM is part of the idle baseline, not the TTS delta
        import numpy as np

        await stt.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    idle_mb = read_used_mb()
    tts = build("tts", option)
    sentences = load_prompts(prompt_set)
    records: list[BenchRecord] = []
    audio_dir = REPORTS / "audio" / option
    try:
        await tts.warmup()
        for s in sentences:
            t0 = time.perf_counter()
            async with VRAMMonitor() as vram:
                chunk = await tts.synthesize(s["text"])
            total = (time.perf_counter() - t0) * 1000
            audio_sec = len(chunk.samples) / chunk.sample_rate
            m = {
                "total_ms": round(total, 1),
                "audio_sec": round(audio_sec, 2),
                "rtf": round((total / 1000) / max(audio_sec, 1e-3), 3),
                "chars": len(s["text"]),
                "ms_per_char": round(total / max(1, len(s["text"])), 1),
                "vram_used_mb": round(vram.peak_mb),
                "vram_delta_mb": round(vram.peak_mb - idle_mb),
                "sample_rate": chunk.sample_rate,
            }
            if save_audio:
                audio_dir.mkdir(parents=True, exist_ok=True)
                sf.write(audio_dir / f"{s['id']}.wav", chunk.samples, chunk.sample_rate)
            if stt:
                hyp = await stt.transcribe(chunk.samples, chunk.sample_rate)
                m["roundtrip_cer"] = round(cer(s["text"], hyp), 3)
                m["roundtrip_text"] = hyp
            records.append(BenchRecord("tts", option, s["id"], m))
    finally:
        await tts.close()
        if stt:
            await stt.close()
    _append(records)
    return records


async def bench_e2e(llm_option: str, tts_option: str, prompt_set: str = "korean_chat", **gen) -> list[BenchRecord]:
    idle_mb = read_used_mb()
    llm, tts = build("llm", llm_option), build("tts", tts_option)
    prompts = load_prompts(prompt_set)
    records: list[BenchRecord] = []
    try:
        await asyncio.gather(llm.warmup(), tts.warmup())
        for p in prompts:
            msgs = _build_messages(p)
            text: list[str] = []
            audio_sec = 0.0
            metrics: dict = {}
            async with VRAMMonitor() as vram:
                async for ev, payload in converse(llm, tts, msgs, **gen):
                    if ev == "text":
                        text.append(payload)
                    elif ev == "audio":
                        audio_sec += len(payload.samples) / payload.sample_rate
                    elif ev == "metrics":
                        metrics = payload
            metrics = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in metrics.items()}
            metrics.update({"audio_sec": round(audio_sec, 2), "vram_used_mb": round(vram.peak_mb), "vram_delta_mb": round(vram.peak_mb - idle_mb), "output_text": "".join(text)})
            records.append(BenchRecord("e2e", f"{llm_option}+{tts_option}", p["id"], metrics))
    finally:
        await asyncio.gather(llm.close(), tts.close())
    _append(records)
    return records
