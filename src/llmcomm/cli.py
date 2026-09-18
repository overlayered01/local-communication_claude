from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.table import Table

from llmcomm.core.params import parse_set, parse_sweep
from llmcomm.core.registry import build, list_configs

app = typer.Typer(help="Local LLM / TTS / STT evaluation testbed", no_args_is_help=True)
bench = typer.Typer(help="Run benchmarks", no_args_is_help=True)
app.add_typer(bench, name="bench")

from llmcomm.core.prompts import DEFAULT_PROMPT, build_messages, describe_prompt, list_prompts
from llmcomm.core.rag import describe_rag, list_rag


@app.command("list")
def list_cmd():
    """List available option configs."""
    for kind in ("llm", "tts", "stt"):
        rprint(f"[bold]{kind}[/]: {', '.join(list_configs(kind)) or '(none)'}")
    rprint(f"[bold]prompt[/]: {', '.join(list_prompts()) or '(none)'}")
    rprint(f"[bold]rag[/]: {', '.join(list_rag()) or '(none)'}")


@app.command("rag")
def rag_cmd(name: str = typer.Argument(...), query: Optional[str] = typer.Option(None, help="이 질문으로 검색해 상위 청크를 보여줌")):
    """Show a RAG option and, with --query, what it retrieves."""
    from llmcomm.core.rag import Retriever, load_rag

    d = describe_rag(name)
    rprint(f"[bold]{name}[/] — {d.get('description', '')}")
    rprint({k: v for k, v in d.items() if k not in ("name", "description", "context_header")})
    if query:
        async def run():
            r = Retriever(load_rag(name))
            await r.index()
            rprint(f"[dim]{len(r.chunks)} chunks, index {r.index_ms:.0f} ms (cached={r.cached})[/]")
            hits, t = await r.retrieve(query)
            rprint(t)
            for h in hits:
                rprint(f"  [cyan]{h.score:.3f}[/] {h.source}: {h.text[:100]}")
            await r.close()
        asyncio.run(run())


@app.command("prompt")
def prompt_cmd(name: str = typer.Argument(DEFAULT_PROMPT)):
    """Show a prompt option: persona, style rules, and the composed system text."""
    d = describe_prompt(name)
    rprint(f"[bold]{name}[/] — {d['description']}")
    rprint(f"[dim]style:[/] {d['style']}")
    rprint("[dim]system:[/]")
    print(d["system"])
    if d["few_shot"]:
        rprint(f"[dim]few_shot:[/] {len(d['few_shot'])} examples")


@app.command()
def chat(llm: str = "ollama_qwen3_14b", tts: Optional[str] = None, prompt: str = DEFAULT_PROMPT, system: Optional[str] = None):
    """Interactive terminal chat; with --tts, sentences are spoken as they stream. --prompt picks a configs/prompt option."""
    from llmcomm.core.types import Message
    from llmcomm.pipeline.player import AudioPlayer
    from llmcomm.pipeline.streaming import converse

    async def run():
        l = build("llm", llm)
        t = build("tts", tts) if tts else None
        history: list[Message] = []
        while True:
            try:
                user = input("\n[you] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user or user in {"/q", "/quit"}:
                break
            history.append(Message("user", user))
            msgs = build_messages(prompt, history, system_override=system)
            print("[bot] ", end="", flush=True)
            reply: list[str] = []
            if t:
                player = AudioPlayer()
                player.start()
                async for ev, p in converse(l, t, msgs):
                    if ev == "text":
                        print(p, end="", flush=True)
                        reply.append(p)
                    elif ev == "audio":
                        await player.put(p)
                    elif ev == "metrics":
                        rprint(f"\n[dim]{p}[/]")
                await player.finish()
            else:
                async for d in l.stream(msgs):
                    print(d, end="", flush=True)
                    reply.append(d)
            history.append(Message("assistant", "".join(reply)))
        await l.close()
        if t:
            await t.close()

    asyncio.run(run())


def _override_sets(set_: Optional[list[str]], sweep: Optional[str]) -> list[dict]:
    """--set k=v (repeatable) plus optional --sweep k=v1,v2 -> one override dict per run."""
    base = parse_set(set_)
    sw = parse_sweep(sweep)
    if not sw:
        return [base]
    k, values = sw
    return [{**base, k: v} for v in values]


SET_HELP = "파라미터 오버라이드 key=value (반복 가능). 이름은 `llmcomm params <kind> <option>`으로 확인"
SWEEP_HELP = "한 파라미터를 여러 값으로 연속 실행: key=v1,v2,v3"


@app.command("params")
def params_cmd(kind: str, option: str):
    """Show tunable parameters of an option: name, type, range, preset value, reload-needed."""
    from llmcomm.core.registry import describe

    d = describe(kind, option)
    t = Table("param", "type", "value", "range / choices", "reload", "description")
    for p in d["params"]:
        rng = ", ".join(p["choices"]) if p["choices"] else (f"{p['min']}..{p['max']}" if p["min"] is not None else "")
        t.add_row(p["name"], p["type"], str(p["value"]), rng, "yes" if p["reload"] else "", p["description"])
    rprint(f"[bold]{kind}/{option}[/] (type: {d['type']})")
    rprint(t)


@bench.command("llm")
def bench_llm_cmd(options: list[str], prompts: str = "korean_chat", repeats: int = 1, max_tokens: int = 200,
                  prompt: list[str] = typer.Option([DEFAULT_PROMPT], help="configs/prompt 옵션 (반복 가능: 프롬프트 비교)"),
                  set_: Optional[list[str]] = typer.Option(None, "--set", help=SET_HELP),
                  sweep: Optional[str] = typer.Option(None, help=SWEEP_HELP)):
    """Benchmark one or more LLM options: TTFT, tokens/s, VRAM, style-rule violations."""
    from llmcomm.bench.runner import bench_llm

    for o in options:
        for pr in prompt:
            for ov in _override_sets(set_, sweep):
                rprint(f"[bold cyan]LLM[/] {o} prompt={pr} {ov or ''}")
                recs = asyncio.run(bench_llm(o, prompts, repeats, overrides=ov, prompt=pr, max_tokens=max_tokens))
                _show(recs, ["ttft_ms", "tokens_per_s_est", "total_ms", "n_sentences", "emoji_count", "style_violations"])


@bench.command("tts")
def bench_tts_cmd(options: list[str], prompts: str = "korean_tts", stt: Optional[str] = None, no_audio: bool = False,
                  set_: Optional[list[str]] = typer.Option(None, "--set", help=SET_HELP),
                  sweep: Optional[str] = typer.Option(None, help=SWEEP_HELP)):
    """Benchmark TTS options: latency, RTF, VRAM, and optional STT round-trip CER."""
    from llmcomm.bench.runner import bench_tts

    for o in options:
        for ov in _override_sets(set_, sweep):
            rprint(f"[bold cyan]TTS[/] {o} {ov or ''}")
            recs = asyncio.run(bench_tts(o, prompts, stt, save_audio=not no_audio, overrides=ov))
            _show(recs, ["total_ms", "rtf", "ms_per_char", "roundtrip_cer", "vram_delta_mb"])


@bench.command("e2e")
def bench_e2e_cmd(llm: str, tts: str, prompts: str = "korean_chat", max_tokens: int = 200, label: Optional[str] = None,
                  set_llm: Optional[list[str]] = typer.Option(None, "--set-llm", help="LLM " + SET_HELP),
                  set_tts: Optional[list[str]] = typer.Option(None, "--set-tts", help="TTS " + SET_HELP),
                  set_pipeline: Optional[list[str]] = typer.Option(None, "--set-pipeline", help="파이프라인: tts_concurrency, min_chars"),
                  prompt: str = typer.Option(DEFAULT_PROMPT, help="configs/prompt 옵션")):
    """Benchmark an LLM+TTS pair through the streaming pipeline: time to first audio.

    --label tags the option name (llm+tts@label) so a pipeline revision is reported as its own row.
    """
    from llmcomm.bench.runner import bench_e2e
    from llmcomm.core.params import PIPELINE_PARAMS, coerce_overrides

    pipe = coerce_overrides(PIPELINE_PARAMS, parse_set(set_pipeline))
    recs = asyncio.run(bench_e2e(llm, tts, prompts, label=label, llm_overrides=parse_set(set_llm), tts_overrides=parse_set(set_tts),
                                 pipeline=pipe, prompt=prompt, max_tokens=max_tokens))
    _show(recs, ["ttft_ms", "first_sentence_ms", "first_audio_ms", "llm_done_ms", "total_ms", "audio_sec", "vram_delta_mb"])


@bench.command("stt")
def bench_stt_cmd(options: list[str], manifest: str = "korean_stt", label: Optional[str] = None,
                  set_: Optional[list[str]] = typer.Option(None, "--set", help=SET_HELP),
                  sweep: Optional[str] = typer.Option(None, help=SWEEP_HELP)):
    """Benchmark STT options on data/stt/<manifest>.jsonl: latency, RTF, CER, VRAM."""
    from llmcomm.bench.runner import bench_stt

    for o in options:
        for ov in _override_sets(set_, sweep):
            rprint(f"[bold cyan]STT[/] {o} {ov or ''}")
            recs = asyncio.run(bench_stt(o, manifest, label, overrides=ov))
            _show(recs, ["latency_ms", "audio_sec", "rtf", "cer", "vram_delta_mb"])
            worst = sorted(recs, key=lambda r: -r.metrics["cer"])[:3]
            for r in worst:
                if r.metrics["cer"] > 0:
                    rprint(f"  [dim]{r.prompt_id}[/] cer={r.metrics['cer']}  ref: {r.metrics['ref_text']}  →  hyp: {r.metrics['output_text']}")


@bench.command("rag")
def bench_rag_cmd(rag: list[str], llm: str = "ollama_qwen3_8b", prompts: Optional[str] = None,
                  prompt: str = typer.Option(DEFAULT_PROMPT, help="configs/prompt 옵션"),
                  no_baseline: bool = typer.Option(False, help="RAG 없는 대조 실행 생략"),
                  set_: Optional[list[str]] = typer.Option(None, "--set", help="LLM " + SET_HELP)):
    """Benchmark RAG options with an LLM: recall@k, answer keyword hit, retrieval/TTFT cost. Baseline (no RAG) included."""
    from llmcomm.bench.runner import bench_rag

    for r in rag:
        rprint(f"[bold cyan]RAG[/] {r} + {llm} prompt={prompt}")
        recs = asyncio.run(bench_rag(r, llm, prompts, prompt=prompt, with_baseline=not no_baseline, overrides=parse_set(set_)))
        _show(recs, ["recall", "keyword_hit", "retrieval_ms", "context_chars", "ttft_ms", "total_ms"])


@app.command("make-stt-set")
def make_stt_set_cmd(tts: list[str] = typer.Option(["edge_sunhi", "supertonic_gpu"], help="TTS voices to synthesize with"),
                     name: str = "korean_stt"):
    """Synthesize an STT evaluation set (korean_tts + korean_chat texts) into data/stt/ with a manifest."""
    from llmcomm.bench.runner import make_stt_set

    path = asyncio.run(make_stt_set(tts, name))
    rprint(f"[green]written[/] {path}")


@app.command("stt-convert")
def stt_convert_cmd(hf_model: str, name: Optional[str] = None, quantization: str = "float16"):
    """Convert a Hugging Face Whisper checkpoint (e.g. a Korean fine-tune) to CTranslate2 under models/ct2/<name>.

    Runs in an isolated uvx environment because the project pins an old transformers for MeloTTS.
    Then reference it from a config: `type: faster_whisper`, `model_size: <name>`.
    """
    import subprocess

    name = name or hf_model.split("/")[-1]
    out = Path("models/ct2") / name
    cmd = ["uvx", "--python", "3.11", "--from", "ctranslate2>=4.4", "--with", "transformers>=4.45,<5", "--with", "torch",
           "--with", "accelerate", "--with", "safetensors", "ct2-transformers-converter", "--model", hf_model,
           "--output_dir", str(out), "--quantization", quantization, "--copy_files", "tokenizer.json", "preprocessor_config.json"]
    rprint(f"[dim]{' '.join(cmd)}[/]")
    subprocess.run(cmd, check=True)
    rprint(f"[green]converted[/] {out}")


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
