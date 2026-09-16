"""Aggregate reports/results.jsonl into a markdown comparison table."""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

KEY_METRICS = {
    "llm": ["ttft_ms", "tokens_per_s_est", "total_ms", "vram_used_mb"],
    "tts": ["total_ms", "rtf", "ms_per_char", "roundtrip_cer", "vram_used_mb"],
    "e2e": ["ttft_ms", "first_audio_ms", "total_ms", "vram_used_mb"],
}


def summarize(path: Path) -> str:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float)):
                by[(r["kind"], r["option"])][k].append(v)
    out: list[str] = ["# Benchmark summary (median per option)"]
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
            cells = [f"{st.median(m[k]):.1f}" if m.get(k) else "-" for k in keys]
            out.append(f"| {o} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)
