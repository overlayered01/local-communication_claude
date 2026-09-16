"""Incremental Korean sentence splitter for LLM token streams.

Feed deltas; get back completed sentences as soon as a terminator is seen.
Uses a fast rule-based path (good enough for streaming); `kss` is used only on flush for leftovers.
Sentences are cleaned for TTS (emoji / markdown removed) before they are emitted.
"""
from __future__ import annotations

import re

try:  # kss import costs ~5s (loads models); pay it at startup, never inside a conversation turn
    import kss as _kss
except Exception:  # pragma: no cover
    _kss = None

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF⭐⭕‼⁉️‍]+"
)
_MARKDOWN = re.compile(r"(\*\*|__|`+|^#+\s*|^\s*[-*]\s+|^\s*\d+\.\s+)", re.M)
_TERMINATORS = re.compile(r"([.!?。！？…]+[\"'”’)\]]*\s*|\n+)")
_HAS_WORD = re.compile(r"[\w가-힣]")
_MIN_CHARS = 6  # avoid emitting "네." alone; merge with next sentence


def clean_for_tts(text: str) -> str:
    """Strip emoji and markdown that TTS engines read aloud or choke on. Keeps sentence punctuation."""
    text = _EMOJI.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


class StreamSentenceSplitter:
    def __init__(self, min_chars: int = _MIN_CHARS):
        self.buf = ""
        self.min_chars = min_chars
        self._pending = ""

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while True:
            m = _TERMINATORS.search(self.buf)
            if not m:
                break
            sent = clean_for_tts(self.buf[: m.end()])
            self.buf = self.buf[m.end():]
            if not sent or not _HAS_WORD.search(sent):
                continue  # empty, emoji-only or punctuation-only fragment
            candidate = (self._pending + " " + sent).strip() if self._pending else sent
            if len(candidate) < self.min_chars:
                self._pending = candidate
                continue
            out.append(candidate)
            self._pending = ""
        return out

    def flush(self) -> list[str]:
        rest = clean_for_tts((self._pending + " " + self.buf).strip())
        self.buf = self._pending = ""
        if not rest or not _HAS_WORD.search(rest):
            return []
        if _kss is None:
            return [rest]
        try:
            parts = [s.strip() for s in _kss.split_sentences(rest) if s.strip()]
            return parts or [rest]
        except Exception:
            return [rest]
