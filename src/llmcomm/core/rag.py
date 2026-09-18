"""RAG as options.

configs/rag/<name>.yaml:
  corpus: <dir name under data/rag/>        documents: *.md / *.txt; "## " headings start a new section
  embedder: ollama                          ollama (any pulled embedding model) | none (BM25 only)
  model: bge-m3                             Ollama embedding model tag
  chunk_chars: 400                          target chunk size (sentence-aligned)
  overlap_sentences: 1
  top_k: 3
  hybrid: false                             also run BM25 (kiwi tokens) and fuse with RRF
  min_score: 0.0                            drop dense hits below this cosine
  context_header: |                         text placed before the retrieved chunks in the system prompt

Retriever.index() chunks the corpus and caches embeddings under models/rag/<corpus>/<model>.npz so repeat
runs cost nothing. retrieve(query) -> [Hit(chunk_id, text, score, source)] and a timing dict.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
RAG_CONFIG = ROOT / "configs" / "rag"
RAG_DATA = ROOT / "data" / "rag"
RAG_CACHE = ROOT / "models" / "rag"

_SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # file#section


@dataclass
class Hit:
    id: str
    text: str
    score: float
    source: str


@dataclass
class RagConfig:
    name: str
    corpus: str
    embedder: str = "ollama"
    model: str = "bge-m3"
    host: str = "http://localhost:11434"
    chunk_chars: int = 400
    overlap_sentences: int = 1
    top_k: int = 3
    hybrid: bool = False
    min_score: float = 0.0
    context_header: str = "다음은 참고 자료입니다. 자료에 있는 내용만 근거로 답하고, 자료에 없으면 모른다고 말하세요."
    description: str = ""


def list_rag() -> list[str]:
    return sorted(p.stem for p in RAG_CONFIG.glob("*.yaml")) if RAG_CONFIG.exists() else []


def load_rag(name: str) -> RagConfig:
    path = RAG_CONFIG / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Available rag: {list_rag()}")
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return RagConfig(name=name, **d)


def list_corpora() -> list[str]:
    return sorted(p.name for p in RAG_DATA.iterdir() if p.is_dir()) if RAG_DATA.exists() else []


# ----------------------------------------------------------------------------------------------- chunking

def chunk_corpus(corpus_dir: Path, chunk_chars: int, overlap_sentences: int) -> list[Chunk]:
    """Split each '## ' section into sentence-aligned chunks of ~chunk_chars with sentence overlap."""
    chunks: list[Chunk] = []
    for f in sorted(list(corpus_dir.glob("*.md")) + list(corpus_dir.glob("*.txt"))):
        text = f.read_text(encoding="utf-8")
        sections = re.split(r"^##\s+", text, flags=re.M)
        for si, sec in enumerate(sections):
            sec = sec.strip()
            if not sec:
                continue
            if si == 0:  # preamble before the first "## ": drop the document title line(s)
                title, body = "", re.sub(r"^#\s.*$", "", sec, flags=re.M).strip()
            else:
                title, _, body = sec.partition("\n")
            title = title.strip()
            sents = [s.strip() for s in _SENT.split(body) if s.strip()]
            if not sents:
                continue
            i, ci = 0, 0
            while i < len(sents):
                buf, j = [], i
                while j < len(sents) and (not buf or sum(len(b) for b in buf) + len(sents[j]) <= chunk_chars):
                    buf.append(sents[j])
                    j += 1
                ctext = (f"[{title}] " if title else "") + " ".join(buf)
                cid = f"{f.stem}#{si}.{ci}"
                chunks.append(Chunk(cid, ctext, f"{f.name}#{title or si}"))
                ci += 1
                if j >= len(sents):
                    break
                i = max(i + 1, j - overlap_sentences)
    return chunks


# ----------------------------------------------------------------------------------------------- embeddings

class OllamaEmbedder:
    def __init__(self, model: str, host: str):
        self.model, self.host = model, host.rstrip("/")
        self._client = httpx.AsyncClient(timeout=120.0)

    async def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 32):
            r = await self._client.post(f"{self.host}/api/embed", json={"model": self.model, "input": texts[i:i + 32]})
            r.raise_for_status()
            out += r.json()["embeddings"]
        arr = np.asarray(out, dtype=np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)

    async def close(self):
        await self._client.aclose()


# ----------------------------------------------------------------------------------------------- BM25 (kiwi)

class BM25:
    def __init__(self, docs: list[str]):
        try:
            from kiwipiepy import Kiwi
            kiwi = Kiwi()
            self._tok = lambda s: [t.form for t in kiwi.tokenize(s) if t.tag[0] in "NVSXM" or t.tag.startswith("SL")]
        except Exception:  # fallback: whitespace + 2-gram
            self._tok = lambda s: re.findall(r"[가-힣A-Za-z0-9]+", s)
        self.docs = [self._tok(d) for d in docs]
        self.avgdl = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        self.df: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.docs)

    def scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        q = self._tok(query)
        s = np.zeros(self.N, dtype=np.float32)
        for t in q:
            if t not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, d in enumerate(self.docs):
                tf = d.count(t)
                if tf:
                    s[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / self.avgdl))
        return s


# ----------------------------------------------------------------------------------------------- retriever

class Retriever:
    def __init__(self, cfg: RagConfig):
        self.cfg = cfg
        self.chunks: list[Chunk] = []
        self.emb: np.ndarray | None = None
        self.bm25: BM25 | None = None
        self._embedder = OllamaEmbedder(cfg.model, cfg.host) if cfg.embedder == "ollama" else None
        self.index_ms = 0.0
        self.cached = False

    @property
    def label(self) -> str:
        return self.cfg.name

    async def index(self) -> None:
        t = time.perf_counter()
        corpus_dir = RAG_DATA / self.cfg.corpus
        if not corpus_dir.exists():
            raise FileNotFoundError(f"corpus {corpus_dir} not found. Available: {list_corpora()}")
        self.chunks = chunk_corpus(corpus_dir, self.cfg.chunk_chars, self.cfg.overlap_sentences)
        if self._embedder:
            key = hashlib.sha1(("\n".join(c.text for c in self.chunks) + self.cfg.model).encode("utf-8")).hexdigest()[:12]
            cache = RAG_CACHE / self.cfg.corpus / f"{self.cfg.model.replace(':', '_')}-{key}.npz"
            if cache.exists():
                self.emb = np.load(cache)["emb"]
                self.cached = True
            else:
                self.emb = await self._embedder.embed([c.text for c in self.chunks])
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, emb=self.emb)
        if self.cfg.hybrid or not self._embedder:
            self.bm25 = BM25([c.text for c in self.chunks])
        self.index_ms = (time.perf_counter() - t) * 1000

    async def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[Hit], dict]:
        k = top_k or self.cfg.top_k
        t0 = time.perf_counter()
        timing: dict = {}
        dense = None
        if self._embedder and self.emb is not None:
            te = time.perf_counter()
            q = (await self._embedder.embed([query]))[0]
            timing["embed_ms"] = round((time.perf_counter() - te) * 1000, 1)
            dense = self.emb @ q
        sparse = self.bm25.scores(query) if self.bm25 else None
        if dense is not None and sparse is not None:  # reciprocal rank fusion
            fused = np.zeros(len(self.chunks), dtype=np.float32)
            for s in (dense, sparse):
                order = np.argsort(-s)
                for rank, i in enumerate(order[:max(20, k * 4)]):
                    fused[i] += 1.0 / (60 + rank)
            score = fused
        else:
            score = dense if dense is not None else sparse
        order = np.argsort(-score)[:k]
        hits = []
        for i in order:
            if dense is not None and sparse is None and float(dense[i]) < self.cfg.min_score:
                continue
            hits.append(Hit(self.chunks[i].id, self.chunks[i].text, float(score[i]), self.chunks[i].source))
        timing["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        timing["context_chars"] = sum(len(h.text) for h in hits)
        return hits, timing

    def context_block(self, hits: list[Hit]) -> str:
        if not hits:
            return ""
        body = "\n".join(f"{i + 1}. {h.text}" for i, h in enumerate(hits))
        return f"{self.cfg.context_header}\n{body}"

    async def close(self):
        if self._embedder:
            await self._embedder.close()


def describe_rag(name: str) -> dict:
    cfg = load_rag(name)
    return {"name": name, **cfg.__dict__}


_RAG_FIELDS = {f for f in RagConfig.__dataclass_fields__ if f != "name"}


def rag_from_dict(name: str, d: dict) -> RagConfig:
    return RagConfig(name=name, **{k: v for k, v in d.items() if k in _RAG_FIELDS})


def save_rag(name: str, d: dict) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError("preset name: letters, digits, _ and - only")
    cfg = rag_from_dict(name, d)
    if not (RAG_DATA / cfg.corpus).exists():
        raise FileNotFoundError(f"corpus {cfg.corpus} not found. Available: {list_corpora()}")
    out = {k: getattr(cfg, k) for k in ("description", "corpus", "embedder", "model", "chunk_chars", "overlap_sentences", "top_k", "hybrid", "min_score", "context_header")}
    RAG_CONFIG.mkdir(parents=True, exist_ok=True)
    from .prompts import dump_yaml

    path = RAG_CONFIG / f"{name}.yaml"
    path.write_text(dump_yaml(out), encoding="utf-8")
    return path


async def list_embedding_models(host: str = "http://localhost:11434") -> list[str]:
    """Embedding-capable models pulled in Ollama (name heuristics; Ollama tags carry no capability flag)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            tags = (await c.get(f"{host}/api/tags")).json().get("models", [])
    except Exception:
        return []
    names = [m["name"] for m in tags]
    return sorted(n for n in names if any(k in n.lower() for k in ("embed", "bge", "e5", "minilm", "arctic")))
