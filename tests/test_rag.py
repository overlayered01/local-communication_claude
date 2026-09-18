import numpy as np
import pytest

from llmcomm.core.prompts import build_messages
from llmcomm.core.rag import BM25, RagConfig, Retriever, chunk_corpus, list_rag, load_rag
from llmcomm.core.types import Message


def test_chunk_corpus_sections_and_overlap(tmp_path):
    (tmp_path / "a.md").write_text("# T\n\n## 첫 절\n문장 하나입니다. 문장 둘입니다. 문장 셋입니다.\n\n## 둘째 절\n짧다.\n", encoding="utf-8")
    chunks = chunk_corpus(tmp_path, chunk_chars=20, overlap_sentences=1)
    assert all(c.source.startswith("a.md#") for c in chunks)
    titles = {c.source.split("#")[1] for c in chunks}
    assert titles == {"첫 절", "둘째 절"}
    first = [c for c in chunks if "첫 절" in c.source]
    assert len(first) >= 2 and first[0].text.startswith("[첫 절]")
    # overlap: the second chunk repeats the last sentence of the first
    assert "문장 둘입니다." in first[0].text and "문장 둘입니다." in first[1].text


def test_bm25_prefers_matching_doc():
    docs = ["알람은 최대 10개까지 저장됩니다.", "보증 기간은 1년입니다.", "블루투스 페어링 모드가 됩니다."]
    s = BM25(docs).scores("알람 몇 개까지 저장돼?")
    assert int(np.argmax(s)) == 0 and s[0] > s[1]


def test_every_rag_config_loads_and_points_at_existing_corpus():
    names = list_rag()
    assert "faq_bm25" in names
    for n in names:
        cfg = load_rag(n)
        assert cfg.corpus and cfg.top_k >= 1
        assert cfg.embedder in ("ollama", "none")


async def test_bm25_only_retriever_end_to_end_without_network():
    r = Retriever(load_rag("faq_bm25"))
    await r.index()
    assert len(r.chunks) > 5 and r.emb is None and r.bm25 is not None
    hits, timing = await r.retrieve("루미 링이 보라색인데 무슨 뜻이야?")
    assert hits and "링 색상" in hits[0].source
    assert timing["retrieval_ms"] < 500 and timing["context_chars"] > 0
    block = r.context_block(hits)
    assert block.startswith(r.cfg.context_header) and "1. " in block
    msgs = build_messages("default", [Message("user", "q")], context=block)
    assert r.cfg.context_header in msgs[0].content and msgs[-1].content == "q"
    await r.close()


def test_rag_config_defaults():
    cfg = RagConfig(name="x", corpus="product_faq")
    assert cfg.model == "bge-m3" and cfg.top_k == 3 and not cfg.hybrid
