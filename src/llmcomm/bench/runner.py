"""Benchmark runners. Each returns a BenchRecord list and appends to reports/results.jsonl."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import soundfile as sf

from llmcomm.core.gpu import VRAMMonitor, read_used_mb
from llmcomm.core.params import effective_params, option_label
from llmcomm.core.prompts import DEFAULT_PROMPT, build_messages, load_prompt, style_metrics
from llmcomm.core.rag import Retriever, load_rag
from llmcomm.core.registry import build
from llmcomm.core.types import BenchRecord, Message
from llmcomm.pipeline.streaming import converse

from .metrics import approx_tokens, cer

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"
PROMPTS = ROOT / "data" / "prompts"
DEFAULT_SYSTEM = load_prompt(DEFAULT_PROMPT).system()  # kept for backward compatibility; prompts are options now


def load_prompts(name: str) -> list[dict]:
    path = PROMPTS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_messages(p: dict, system: str | None = None, prompt: str = DEFAULT_PROMPT) -> list[Message]:
    """Prompt rows are either {"user": ...} or multi-turn {"messages": [...]}.
    The system message comes from the prompt option (configs/prompt/<prompt>.yaml); `system` overrides it verbatim."""
    if "messages" in p:
        hist = [Message(m["role"], m["content"]) for m in p["messages"]]
    else:
        hist = [Message("user", p["user"])]
    return build_messages(prompt, hist, system_override=system)


def _append(records: list[BenchRecord], out: Path | None = None) -> None:
    out = out or REPORTS / "results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


async def bench_llm(option: str, prompt_set: str = "korean_chat", repeats: int = 1, system: str | None = None,
                    overrides: dict | None = None, prompt: str = DEFAULT_PROMPT, **gen) -> list[BenchRecord]:
    """`prompt` is a configs/prompt option; non-default prompts get an `@prompt=<name>` suffix on the option name
    so prompt variants are separate rows, and every record carries rule-based style metrics."""
    idle_mb = read_used_mb()
    llm = build("llm", option, overrides)
    pcfg = load_prompt(prompt)
    prompts = load_prompts(prompt_set)
    records: list[BenchRecord] = []
    meta = {"params": effective_params(llm), "overrides": llm.overrides, "prompt": prompt, "system": system or pcfg.system()}
    opt_name = llm.label + (f"@prompt={prompt}" if prompt != DEFAULT_PROMPT else "")
    try:
        await llm.warmup()
        for p in prompts:
            for rep in range(repeats):
                msgs = _build_messages(p, system, prompt)
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
                records.append(BenchRecord("llm", opt_name, p["id"], {
                    **style_metrics(text, pcfg.style),
                    "ttft_ms": round(ttft or 0, 1),
                    "total_ms": round(total, 1),
                    "output_tokens_est": toks,
                    "tokens_per_s_est": round(toks / (gen_ms / 1000), 1),
                    "vram_used_mb": round(vram.peak_mb),
                    "vram_delta_mb": round(vram.peak_mb - idle_mb),
                    "output_text": text,
                }, meta={**meta, "rep": rep}))
    finally:
        await llm.close()
    _append(records)
    return records


async def bench_tts(option: str, prompt_set: str = "korean_tts", stt_option: str | None = None, save_audio: bool = True,
                    overrides: dict | None = None) -> list[BenchRecord]:
    stt = build("stt", stt_option) if stt_option else None
    if stt:  # load the STT model first so its VRAM is part of the idle baseline, not the TTS delta
        import numpy as np

        await stt.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    idle_mb = read_used_mb()
    tts = build("tts", option, overrides)
    sentences = load_prompts(prompt_set)
    records: list[BenchRecord] = []
    meta = {"params": effective_params(tts), "overrides": tts.overrides}
    audio_dir = REPORTS / "audio" / tts.label
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
            records.append(BenchRecord("tts", tts.label, s["id"], m, meta=meta))
    finally:
        await tts.close()
        if stt:
            await stt.close()
    _append(records)
    return records


async def bench_e2e(llm_option: str, tts_option: str, prompt_set: str = "korean_chat", label: str | None = None,
                    llm_overrides: dict | None = None, tts_overrides: dict | None = None, pipeline: dict | None = None,
                    prompt: str = DEFAULT_PROMPT, **gen) -> list[BenchRecord]:
    """`label` is appended to the option name (e.g. "@v2") so pipeline revisions stay separate rows in the report.
    `pipeline` holds converse() knobs (tts_concurrency, min_chars)."""
    idle_mb = read_used_mb()
    llm, tts = build("llm", llm_option, llm_overrides), build("tts", tts_option, tts_overrides)
    pipeline = pipeline or {}
    pcfg = load_prompt(prompt)
    meta = {"params": {"llm": effective_params(llm), "tts": effective_params(tts), "pipeline": pipeline},
            "overrides": {"llm": llm.overrides, "tts": tts.overrides}, "prompt": prompt}
    prompts = load_prompts(prompt_set)
    records: list[BenchRecord] = []
    try:
        await asyncio.gather(llm.warmup(), tts.warmup())
        for p in prompts:
            msgs = _build_messages(p, prompt=prompt)
            text: list[str] = []
            audio_sec = 0.0
            metrics: dict = {}
            async with VRAMMonitor() as vram:
                async for ev, payload in converse(llm, tts, msgs, **pipeline, **gen):
                    if ev == "text":
                        text.append(payload)
                    elif ev == "audio":
                        audio_sec += len(payload.samples) / payload.sample_rate
                    elif ev == "metrics":
                        metrics = payload
            metrics = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in metrics.items()}
            metrics.update({"audio_sec": round(audio_sec, 2), "vram_used_mb": round(vram.peak_mb), "vram_delta_mb": round(vram.peak_mb - idle_mb), "output_text": "".join(text)})
            metrics.update(style_metrics("".join(text), pcfg.style))
            name = f"{llm.label}+{tts.label}" + (option_label("", pipeline)[0:] if pipeline else "") + (f"@prompt={prompt}" if prompt != DEFAULT_PROMPT else "") + (f"@{label}" if label else "")
            records.append(BenchRecord("e2e", name, p["id"], metrics, meta=meta))
    finally:
        await asyncio.gather(llm.close(), tts.close())
    _append(records)
    return records


# ---------------------------------------------------------------------------------------------
# STT

STT_DATA = ROOT / "data" / "stt"


def load_stt_manifest(name: str) -> list[dict]:
    """data/stt/<name>.jsonl rows: {"id", "audio" (path relative to data/stt), "text", optional "voice"}."""
    path = STT_DATA / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build one with `llmcomm make-stt-set` or record via the web tester.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def make_stt_set(tts_options: list[str], name: str = "korean_stt", prompt_sets: tuple[str, ...] = ("korean_tts", "korean_chat")) -> Path:
    """Synthesize an STT evaluation set: every text in the prompt sets, spoken by each TTS voice.

    Synthetic speech is clean and consistent, so it measures the engine's Korean modeling rather than
    robustness to real microphones. Add human recordings (web tester mic saves them under data/stt/audio/mic)
    for the numbers that matter for the product.
    """
    rows: list[dict] = []
    texts = []
    for ps in prompt_sets:
        for p in load_prompts(ps):
            t = p.get("text") or p.get("user")
            if t:
                texts.append((f"{ps}_{p['id']}", t))
    for opt in tts_options:
        tts = build("tts", opt)
        out_dir = STT_DATA / "audio" / opt
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            await tts.warmup()
            for tid, text in texts:
                chunk = await tts.synthesize(text)
                rel = f"audio/{opt}/{tid}.wav"
                sf.write(STT_DATA / rel, chunk.samples, chunk.sample_rate)
                rows.append({"id": f"{opt}/{tid}", "audio": rel, "text": text, "voice": opt})
        finally:
            await tts.close()
    manifest = STT_DATA / f"{name}.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return manifest


async def bench_stt(option: str, manifest: str = "korean_stt", label: str | None = None, overrides: dict | None = None) -> list[BenchRecord]:
    """Per clip: latency, real-time factor, CER vs reference, VRAM delta. Whole-clip (non-streaming) decoding."""
    idle_mb = read_used_mb()
    stt = build("stt", option, overrides)
    rows = load_stt_manifest(manifest)
    records: list[BenchRecord] = []
    opt_name = stt.label + (f"@{label}" if label else "")
    params = effective_params(stt)
    try:
        await stt.warmup()
        for r in rows:
            samples, sr = sf.read(STT_DATA / r["audio"], dtype="float32", always_2d=True)
            samples = samples.mean(axis=1)
            audio_sec = len(samples) / sr
            t0 = time.perf_counter()
            async with VRAMMonitor() as vram:
                hyp = await stt.transcribe(samples, sr)
            latency = (time.perf_counter() - t0) * 1000
            records.append(BenchRecord("stt", opt_name, r["id"], {
                "latency_ms": round(latency, 1),
                "audio_sec": round(audio_sec, 2),
                "rtf": round((latency / 1000) / max(audio_sec, 1e-3), 3),
                "cer": round(cer(r["text"], hyp), 3),
                "vram_used_mb": round(vram.peak_mb),
                "vram_delta_mb": round(vram.peak_mb - idle_mb),
                "ref_text": r["text"],
                "output_text": hyp,
            }, meta={"voice": r.get("voice", ""), "streaming_capable": getattr(stt, "streaming", False), "params": params, "overrides": stt.overrides}))
    finally:
        await stt.close()
    _append(records)
    return records


# ---------------------------------------------------------------------------------------------
# RAG

def _keyword_hit(text: str, keywords: list[str]) -> bool:
    t = text.replace(" ", "")
    return any(k.replace(" ", "") in t for k in keywords)


async def bench_rag(rag_option: str, llm_option: str, prompt_set: str | None = None, prompt: str = DEFAULT_PROMPT,
                    with_baseline: bool = True, overrides: dict | None = None, **gen) -> list[BenchRecord]:
    """Retrieval quality + answer grounding + latency cost of a RAG option with a given LLM.

    Prompt rows: {"id","user","must_include_any":[...],"gold_source": "file#section" | null}.
    Per question: recall (gold section in top-k), retrieval_ms, context_chars, answer keyword hit, TTFT/total.
    With `with_baseline`, the same question is also asked without context (option name `<llm>@rag=none`) so the
    TTFT delta and the grounding gain are directly comparable.
    """
    cfg = load_rag(rag_option)
    prompt_set = prompt_set or f"korean_rag_{cfg.corpus}"
    rows = load_prompts(prompt_set)
    retriever = Retriever(cfg)
    idle_mb = read_used_mb()
    llm = build("llm", llm_option, overrides)
    pcfg = load_prompt(prompt)
    records: list[BenchRecord] = []
    try:
        await retriever.index()
        await llm.warmup()
        await retriever.retrieve("워밍업")
        for r in rows:
            variants = [("rag", rag_option)] + ([("none", None)] if with_baseline else [])
            for mode, _ in variants:
                hits, timing = ([], {"retrieval_ms": 0.0, "context_chars": 0})
                if mode == "rag":
                    hits, timing = await retriever.retrieve(r["user"])
                msgs = build_messages(pcfg, [Message("user", r["user"])], context=retriever.context_block(hits) if hits else None)
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
                gold = r.get("gold_source")
                sources = [h.source.split(".md")[0] + h.source[h.source.index("#"):] if "#" in h.source else h.source for h in hits]
                recall = None if gold is None else float(any(gold == s or gold.split("#")[-1] == s.split("#")[-1] for s in sources))
                metrics = {
                    "retrieval_ms": round(timing.get("retrieval_ms", 0.0), 1),
                    "embed_ms": round(timing.get("embed_ms", 0.0), 1),
                    "context_chars": timing.get("context_chars", 0),
                    "prompt_chars": sum(len(m.content) for m in msgs),
                    "recall": recall,
                    "keyword_hit": float(_keyword_hit(text, r.get("must_include_any", []))),
                    "ttft_ms": round(ttft or 0, 1),
                    "total_ms": round(total, 1),
                    "e2e_ms": round(total + timing.get("retrieval_ms", 0.0), 1),
                    "vram_delta_mb": round(vram.peak_mb - idle_mb),
                    "output_text": text,
                    **style_metrics(text, pcfg.style),
                }
                name = f"{llm.label}@rag={rag_option if mode == 'rag' else 'none:' + cfg.corpus}" + (f"@prompt={prompt}" if prompt != DEFAULT_PROMPT else "")
                records.append(BenchRecord("rag", name, r["id"], metrics, meta={
                    "rag": cfg.__dict__ if mode == "rag" else None, "hits": [h.source for h in hits], "gold": gold,
                    "params": effective_params(llm), "prompt": prompt}))
    finally:
        await llm.close()
        await retriever.close()
    _append(records)
    return records
