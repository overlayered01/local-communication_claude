from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.table import Table

from llmcomm.core.registry import build, list_configs

app = typer.Typer(help="Local LLM / TTS / STT evaluation testbed", no_args_is_help=True)
bench = typer.Typer(help="Run benchmarks", no_args_is_help=True)
app.add_typer(bench, name="bench")

DEFAULT_SYSTEM = "당신은 친근한 한국어 대화 상대입니다. 짧고 자연스럽게 답하세요."


@app.command("list")
def list_cmd():
    """List available option configs."""
    for kind in ("llm", "tts", "stt"):
        rprint(f"[bold]{kind}[/]: {', '.join(list_configs(kind)) or '(none)'}")


@app.command()
def chat(llm: str = "ollama_qwen3_14b", tts: Optional[str] = None, system: str = DEFAULT_SYSTEM):
    """Interactive terminal chat; with --tts, sentences are spoken as they stream."""
    from llmcomm.core.types import Message
    from llmcomm.pipeline.player import AudioPlayer
    from llmcomm.pipeline.streaming import converse

    async def run():
        l = build("llm", llm)
        t = build("tts", tts) if tts else None
        history = [Message("system", system)]
        while True:
            try:
                user = input("\n[you] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user or user in {"/q", "/quit"}:
                break
            history.append(Message("user", user))
            print("[bot] ", end="", flush=True)
            reply: list[str] = []
            if t:
                player = AudioPlayer()
                player.start()
                async for ev, p in converse(l, t, history):
                    if ev == "text":
                        print(p, end="", flush=True)
                        reply.append(p)
                    elif ev == "audio":
                        await player.put(p)
                    elif ev == "metrics":
                        rprint(f"\n[dim]{p}[/]")
                await player.finish()
            else:
                async for d in l.stream(history):
                    print(d, end="", flush=True)
                    reply.append(d)
            history.append(Message("assistant", "".join(reply)))
        await l.close()
        if t:
            await t.close()

    asyncio.run(run())


@bench.command("llm")
def bench_llm_cmd(options: list[str], prompts: str = "korean_chat", repeats: int = 1, max_tokens: int = 200):
    """Benchmark one or more LLM options: TTFT, tokens/s, VRAM."""
    from llmcomm.bench.runner import bench_llm

    for o in options:
        rprint(f"[bold cyan]LLM[/] {o}")
        recs = asyncio.run(bench_llm(o, prompts, repeats, max_tokens=max_tokens))
        _show(recs, ["ttft_ms", "tokens_per_s_est", "total_ms", "vram_delta_mb"])


@bench.command("tts")
def bench_tts_cmd(options: list[str], prompts: str = "korean_tts", stt: Optional[str] = None, no_audio: bool = False):
    """Benchmark TTS options: latency, RTF, VRAM, and optional STT round-trip CER."""
    from llmcomm.bench.runner import bench_tts

    for o in options:
        rprint(f"[bold cyan]TTS[/] {o}")
        recs = asyncio.run(bench_tts(o, prompts, stt, save_audio=not no_audio))
        _show(recs, ["total_ms", "rtf", "ms_per_char", "roundtrip_cer", "vram_delta_mb"])


@bench.command("e2e")
def bench_e2e_cmd(llm: str, tts: str, prompts: str = "korean_chat", max_tokens: int = 200):
    """Benchmark an LLM+TTS pair through the streaming pipeline: time to first audio."""
    from llmcomm.bench.runner import bench_e2e

    recs = asyncio.run(bench_e2e(llm, tts, prompts, max_tokens=max_tokens))
    _show(recs, ["ttft_ms", "first_sentence_ms", "first_audio_ms", "llm_done_ms", "total_ms", "audio_sec", "vram_delta_mb"])


@app.command()
def report(results: Path = Path("reports/results.jsonl"), out: Path = Path("reports/summary.md")):
    """Aggregate results into a markdown comparison table."""
    from llmcomm.bench.report import summarize

    md = summarize(results)
    out.write_text(md, encoding="utf-8")
    print(md)
    rprint(f"\n[green]written[/] {out}")


@app.command()
def judge(judge: str = "ollama_qwen3_14b", results: Path = Path("reports/results.jsonl")):
    """Score LLM outputs in results.jsonl with an LLM judge (naturalness/korean/relevance/brevity, 1-5)."""
    from llmcomm.bench.judge import judge_llm_results

    summary = asyncio.run(judge_llm_results(judge, results))
    t = Table("option", "n", "naturalness", "korean", "relevance", "brevity", "mean")
    for opt, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean"]):
        t.add_row(opt, str(s["n"]), *(str(s[k]) for k in ("naturalness", "korean", "relevance", "brevity", "mean")))
    rprint(t)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8080):
    """Run the web tester (FastAPI + WebSocket)."""
    import uvicorn

    uvicorn.run("llmcomm.server.app:app", host=host, port=port)


def _show(recs, keys):
    t = Table(*(["prompt"] + keys))
    for r in recs:
        t.add_row(r.prompt_id, *(str(r.metrics.get(k, "-")) for k in keys))
    rprint(t)


if __name__ == "__main__":
    app()
