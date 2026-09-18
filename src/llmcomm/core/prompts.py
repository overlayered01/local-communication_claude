"""System prompts as options.

configs/prompt/<name>.yaml:
  description: 한 줄 설명
  persona: |            free text: who the assistant is
  style:                machine-checkable rules; each one is appended to the system prompt as an instruction
    register: banmal | jondae | free
    max_sentences: 2     (0 = no limit)
    no_emoji: true
    no_markdown: true
    language: ko
  few_shot:             optional [{user, assistant}] examples inserted after the system message
  extra: |              optional free-text rules appended verbatim

`compose_system()` turns persona + style into the final system text, so the rules the model is told and the
rules `style_metrics()` checks are always the same. `style_metrics()` scores a reply against the style
without an LLM judge: sentence count, emoji/markdown count, register violations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .types import Message

PROMPT_ROOT = Path(__file__).resolve().parents[3] / "configs" / "prompt"
DEFAULT_PROMPT = "default"

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF⭐⭕‼⁉]")
_MARKDOWN = re.compile(r"(\*\*|__|`|^#+\s|^\s*[-*]\s+|^\s*\d+\.\s+)", re.M)
_SENT_END = re.compile(r"[.!?。！？…]+[\"'”’)\]]*\s*|\n+")
# polite (존댓말) sentence endings; anything else is treated as casual (반말)
_POLITE = ("요", "습니다", "습니까", "세요", "십니다", "십니까", "니다", "네요", "군요", "죠", "셨어", "십쇼", "십시오", "ㅂ니다")
_HAS_HANGUL = re.compile(r"[가-힣]")


@dataclass
class Style:
    register: str = "free"  # banmal | jondae | free
    max_sentences: int = 0
    no_emoji: bool = False
    no_markdown: bool = False
    language: str = "ko"

    def rules(self) -> list[str]:
        r: list[str] = []
        if self.register == "banmal":
            r.append("항상 친한 친구에게 말하듯 반말로만 말해. 존댓말('~요', '~습니다')은 쓰지 마.")
        elif self.register == "jondae":
            r.append("항상 정중한 존댓말('~요', '~습니다')로만 말하세요. 반말은 쓰지 마세요.")
        if self.max_sentences:
            r.append(f"답은 {self.max_sentences}문장 이내로 짧게 하세요." if self.register != "banmal" else f"답은 {self.max_sentences}문장 이내로 짧게 해.")
        if self.no_emoji:
            r.append("이모지와 이모티콘은 절대 쓰지 마세요." if self.register != "banmal" else "이모지와 이모티콘은 절대 쓰지 마.")
        if self.no_markdown:
            r.append("마크다운, 목록, 굵은 글씨, 제목 표시는 쓰지 말고 말로만 답하세요." if self.register != "banmal" else "마크다운, 목록, 굵은 글씨는 쓰지 말고 말로만 답해.")
        if self.language == "ko":
            r.append("한국어로만 답하세요. 영어나 한자를 섞지 마세요." if self.register != "banmal" else "한국어로만 답해. 영어나 한자를 섞지 마.")
        return r


@dataclass
class PromptConfig:
    name: str
    persona: str
    style: Style = field(default_factory=Style)
    few_shot: list[dict] = field(default_factory=list)
    extra: str = ""
    description: str = ""

    def system(self) -> str:
        return compose_system(self.persona, self.style, self.extra)


def compose_system(persona: str, style: Style, extra: str = "") -> str:
    parts = [persona.strip()]
    rules = style.rules()
    if rules:
        parts.append("\n".join(f"- {r}" for r in rules))
    if extra.strip():
        parts.append(extra.strip())
    return "\n\n".join(parts)


def list_prompts() -> list[str]:
    return sorted(p.stem for p in PROMPT_ROOT.glob("*.yaml"))


def load_prompt(name: str) -> PromptConfig:
    path = PROMPT_ROOT / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Available prompts: {list_prompts()}")
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    st = d.get("style") or {}
    return PromptConfig(name=name, persona=d.get("persona", ""), style=Style(**st), few_shot=d.get("few_shot") or [],
                        extra=d.get("extra", "") or "", description=d.get("description", ""))


def describe_prompt(name: str) -> dict:
    p = load_prompt(name)
    return {"name": name, "description": p.description, "persona": p.persona, "style": p.style.__dict__,
            "few_shot": p.few_shot, "extra": p.extra, "system": p.system()}


def build_messages(prompt: PromptConfig | str | None, history: list[Message], system_override: str | None = None,
                   context: str | None = None) -> list[Message]:
    """System (from the prompt option or a free-text override) + optional RAG context block + few-shot +
    conversation history (no system in it)."""
    cfg = load_prompt(prompt) if isinstance(prompt, str) else (prompt or load_prompt(DEFAULT_PROMPT))
    system = system_override if system_override is not None else cfg.system()
    if context:
        system = f"{system}\n\n{context}"
    msgs = [Message("system", system)]
    for ex in cfg.few_shot:
        msgs.append(Message("user", ex["user"]))
        msgs.append(Message("assistant", ex["assistant"]))
    msgs += [m for m in history if m.role != "system"]
    return msgs


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    idx = 0
    for m in _SENT_END.finditer(text):
        s = text[idx:m.end()].strip()
        if s:
            out.append(s)
        idx = m.end()
    tail = text[idx:].strip()
    if tail:
        out.append(tail)
    return out


def is_polite(sentence: str) -> bool | None:
    """True=존댓말, False=반말, None=cannot tell (no Hangul / too short)."""
    s = re.sub(r"[\s\W_]+$", "", sentence.strip())
    s = _EMOJI.sub("", s).rstrip()
    if not _HAS_HANGUL.search(s) or len(s) < 2:
        return None
    return s.endswith(_POLITE)


def style_metrics(text: str, style: Style) -> dict:
    sents = split_sentences(text)
    judged = [is_polite(s) for s in sents]
    polite = sum(1 for j in judged if j is True)
    casual = sum(1 for j in judged if j is False)
    emoji = len(_EMOJI.findall(text))
    markdown = len(_MARKDOWN.findall(text))
    latin = len(re.findall(r"[A-Za-z]{2,}", text))
    hanja = len(re.findall(r"[一-鿿]", text))
    viol = 0
    if style.register == "banmal":
        viol += polite
    elif style.register == "jondae":
        viol += casual
    if style.max_sentences and len(sents) > style.max_sentences:
        viol += 1
    if style.no_emoji and emoji:
        viol += 1
    if style.no_markdown and markdown:
        viol += 1
    if style.language == "ko" and (hanja or latin > 2):
        viol += 1
    return {
        "chars": len(text),
        "n_sentences": len(sents),
        "polite_sentences": polite,
        "casual_sentences": casual,
        "emoji_count": emoji,
        "markdown_count": markdown,
        "latin_words": latin,
        "hanja_chars": hanja,
        "style_violations": viol,
    }


def prompt_from_dict(name: str, d: dict) -> PromptConfig:
    """Build a PromptConfig from a request/UI dict (same shape as the YAML)."""
    st = {k: v for k, v in (d.get("style") or {}).items() if k in Style.__dataclass_fields__}
    return PromptConfig(name=name, persona=d.get("persona", "") or "", style=Style(**st), few_shot=d.get("few_shot") or [],
                        extra=d.get("extra", "") or "", description=d.get("description", "") or "")


class _BlockDumper(yaml.SafeDumper):
    """Multiline strings as `|` blocks so saved presets stay human-editable."""


def _str_presenter(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data.rstrip("\n") + "\n", style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockDumper.add_representer(str, _str_presenter)


def dump_yaml(obj: dict) -> str:
    return yaml.dump(obj, Dumper=_BlockDumper, allow_unicode=True, sort_keys=False, width=1000)


def save_prompt(name: str, d: dict) -> Path:
    """Write configs/prompt/<name>.yaml from a dict; validates by round-tripping through PromptConfig."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError("preset name: letters, digits, _ and - only")
    cfg = prompt_from_dict(name, d)
    out = {"description": cfg.description, "persona": cfg.persona.rstrip() + "\n", "style": cfg.style.__dict__}
    if cfg.few_shot:
        out["few_shot"] = cfg.few_shot
    if cfg.extra.strip():
        out["extra"] = cfg.extra
    PROMPT_ROOT.mkdir(parents=True, exist_ok=True)
    path = PROMPT_ROOT / f"{name}.yaml"
    path.write_text(dump_yaml(out), encoding="utf-8")
    return path
