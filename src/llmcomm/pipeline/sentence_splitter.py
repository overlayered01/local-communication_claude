"""Incremental Korean sentence splitter for LLM token streams.

Feed deltas; get back completed sentences as soon as a terminator is seen.
Uses a fast rule-based path (good enough for streaming); `kss` is used only on flush for leftovers.
"""
from __future__ import annotations

import re

_TERMINATORS = re.compile(r"([.!?。！？…]+[\"'”’)\]]*\s*|\n+)")
_MIN_CHARS = 6  # avoid emitting "네." alone; merge with next sentence


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
            sent = self.buf[: m.end()].strip()
            self.buf = self.buf[m.end():]
            if not sent:
                continue
            candidate = (self._pending + " " + sent).strip() if self._pending else sent
            if len(candidate) < self.min_chars:
                self._pending = candidate
                continue
            out.append(candidate)
            self._pending = ""
        return out

    def flush(self) -> list[str]:
        rest = (self._pending + " " + self.buf).strip()
        self.buf = self._pending = ""
        if not rest:
            return []
        try:
            import kss

            parts = [s.strip() for s in kss.split_sentences(rest) if s.strip()]
            return parts or [rest]
        except Exception:
            return [rest]
