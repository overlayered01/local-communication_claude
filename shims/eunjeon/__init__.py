"""Drop-in replacement for the `eunjeon` package (Mecab-ko wrapper) on Windows.

`eunjeon` needs a Visual Studio build and mecab-ko binaries; g2pkk (used by MeloTTS Korean) only calls
`Mecab().pos(text) -> list[(morpheme, tag)]`. Kiwi ships prebuilt wheels and uses the same Sejong tagset
(NNG, JKS, EF, ...), so it is a good functional substitute.

Activated by adding the `shims/` directory to sys.path (see llmcomm.tts.melotts).
"""
from __future__ import annotations


class Mecab:
    def __init__(self, *args, **kwargs):
        from kiwipiepy import Kiwi

        self._kiwi = Kiwi()

    def pos(self, text: str, **kwargs) -> list[tuple[str, str]]:
        return [(t.form, t.tag) for t in self._kiwi.tokenize(text)]

    def morphs(self, text: str) -> list[str]:
        return [f for f, _ in self.pos(text)]

    def nouns(self, text: str) -> list[str]:
        return [f for f, t in self.pos(text) if t.startswith("N")]
