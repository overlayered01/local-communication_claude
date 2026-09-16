from __future__ import annotations

import re

_STRIP = re.compile(r"[\s.,!?~'\"…·]")


def _norm(s: str) -> str:
    return _STRIP.sub("", s)


def cer(ref: str, hyp: str) -> float:
    """Character error rate on whitespace/punctuation-stripped Korean text."""
    r, h = _norm(ref), _norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def approx_tokens(text: str) -> int:
    """Rough token estimate when the backend doesn't report counts (Korean ~1 token per 1.5 chars)."""
    return max(1, int(len(text) / 1.5))
