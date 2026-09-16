"""LLM-as-judge scoring of LLM benchmark outputs in reports/results.jsonl.

Any registered LLM option can act as judge. Prefer a model that is NOT one of the candidates,
or at least the largest available one; note self-preference bias when reading results.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from llmcomm.core.registry import build
from llmcomm.core.types import Message

from .runner import PROMPTS, REPORTS

RUBRIC = """당신은 한국어 대화형 AI의 응답을 평가하는 심사관입니다. 아래 사용자 발화와 AI 응답을 보고 네 항목을 1~5점으로 채점하세요.

- naturalness: 실제 사람이 대화하듯 자연스러운가 (어색한 번역투, 기계적 문장 감점)
- korean: 한국어 문법, 어휘, 존댓말/반말 일관성이 올바른가
- relevance: 사용자의 의도와 맥락에 맞게 답했는가
- brevity: 음성 대화에 적합한 길이인가 (장황함, 목록/마크다운 사용 감점)

반드시 아래 JSON 한 줄만 출력하세요. 설명은 쓰지 마세요.
{"naturalness": n, "korean": n, "relevance": n, "brevity": n}"""


def _load_prompt_map() -> dict[str, dict]:
    m = {}
    for f in PROMPTS.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                m[d["id"]] = d
    return m


def _user_text(p: dict) -> str:
    if "messages" in p:
        return "\n".join(f"[{m['role']}] {m['content']}" for m in p["messages"])
    return p["user"]


def _parse(s: str) -> dict | None:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    m = re.search(r"\{.*?\}", s, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {k: float(d[k]) for k in ("naturalness", "korean", "relevance", "brevity")}
    except Exception:
        return None


async def judge_llm_results(judge_option: str, results: Path | None = None, out: Path | None = None) -> dict[str, dict[str, float]]:
    results = results or REPORTS / "results.jsonl"
    out = out or REPORTS / "judge.jsonl"
    prompts = _load_prompt_map()
    rows = [json.loads(l) for l in results.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["kind"] == "llm" and r["metrics"].get("output_text")]
    judge = build("llm", judge_option)
    scores: dict[str, list[dict]] = defaultdict(list)
    try:
        with out.open("a", encoding="utf-8") as f:
            for r in rows:
                p = prompts.get(r["prompt_id"], {"user": "(unknown)"})
                convo = f"### 사용자 발화\n{_user_text(p)}\n\n### AI 응답\n{r['metrics']['output_text']}"
                raw = await judge.complete([Message("system", RUBRIC), Message("user", convo)], max_tokens=200, temperature=0.0)
                parsed = _parse(raw)
                rec = {"judge": judge_option, "option": r["option"], "prompt_id": r["prompt_id"], "scores": parsed, "raw": raw if parsed is None else None}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if parsed:
                    scores[r["option"]].append(parsed)
    finally:
        await judge.close()
    summary: dict[str, dict[str, float]] = {}
    for opt, lst in scores.items():
        summary[opt] = {k: round(sum(d[k] for d in lst) / len(lst), 2) for k in lst[0]}
        summary[opt]["n"] = len(lst)
        summary[opt]["mean"] = round(sum(v for k, v in summary[opt].items() if k not in ("n",)) / 4, 2)
    return summary
