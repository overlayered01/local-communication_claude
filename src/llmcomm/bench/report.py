"""Aggregate reports/results.jsonl (+ judge.jsonl) into a markdown comparison report."""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

KEY_METRICS = {
    "llm": ["ttft_ms", "tokens_per_s_est", "total_ms", "vram_delta_mb"],
    "tts": ["total_ms", "rtf", "ms_per_char", "roundtrip_cer", "vram_delta_mb"],
    "e2e": ["ttft_ms", "first_sentence_ms", "first_audio_ms", "llm_done_ms", "total_ms", "audio_sec", "vram_delta_mb"],
}
JUDGE_KEYS = ["naturalness", "korean", "relevance", "brevity"]


def _p95(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def _cell(vals: list[float] | None, key: str) -> str:
    if not vals:
        return "-"
    med = st.median(vals)
    if key.endswith("_ms") and len(vals) > 1:
        return f"{med:.0f} ({_p95(vals):.0f})"
    return f"{med:.1f}" if isinstance(med, float) and med != int(med) else f"{med:.0f}"


def summarize_judge(path: Path) -> str:
    if not path.exists():
        return ""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by: dict[str, list[dict]] = defaultdict(list)
    judges: set[str] = set()
    for r in rows:
        if r.get("scores"):
            by[r["option"]].append(r["scores"])
            judges.add(r["judge"])
    if not by:
        return ""
    out = [
        f"\n## LLM quality (judge: {', '.join(sorted(judges))}; 1-5, mean)\n",
        "| option | n | " + " | ".join(JUDGE_KEYS) + " | mean |",
        "|---|---|" + "---|" * (len(JUDGE_KEYS) + 1),
    ]
    ranked = []
    for opt, lst in by.items():
        means = {k: sum(d[k] for d in lst) / len(lst) for k in JUDGE_KEYS}
        ranked.append((sum(means.values()) / len(JUDGE_KEYS), opt, len(lst), means))
    for mean, opt, n, means in sorted(ranked, reverse=True):
        out.append(f"| {opt} | {n} | " + " | ".join(f"{means[k]:.2f}" for k in JUDGE_KEYS) + f" | {mean:.2f} |")
    return "\n".join(out)


def summarize(path: Path) -> str:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float)):
                by[(r["kind"], r["option"])][k].append(v)
    out: list[str] = ["# Benchmark summary", "", "median per option; latency columns also show p95 in parentheses"]
    for kind, keys in KEY_METRICS.items():
        opts = sorted(o for (k, o) in by if k == kind)
        if not opts:
            continue
        out.append(f"\n## {kind.upper()}\n")
        out.append("| option | n | " + " | ".join(keys) + " |")
        out.append("|---|---|" + "---|" * len(keys))
        for o in opts:
            m = by[(kind, o)]
            n = max((len(v) for v in m.values()), default=0)
            out.append(f"| {o} | {n} | " + " | ".join(_cell(m.get(k), k) for k in keys) + " |")
    judge_md = summarize_judge(path.parent / "judge.jsonl")
    return "\n".join(out) + ("\n" + judge_md if judge_md else "")
