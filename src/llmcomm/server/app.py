"""FastAPI server: WebSocket /ws/chat streams text + PCM audio for any llm/tts option pair.

Client sends JSON: {"llm": "...", "tts": "...", "messages": [{"role": ..., "content": ...}]}
  or, for voice input, {"type":"audio_in","llm","tts","stt","sample_rate":16000,"messages":[...]} followed by one
  binary PCM16 mono frame; the server transcribes it, sends {"type":"transcript","text","ms"}, appends the text
  as the user turn and continues as a normal chat turn. Recordings are saved under data/stt/audio/mic/ with a
  mic_log.jsonl so they can be corrected into a human STT evaluation set.
Server sends:
  {"type":"status","stage":"loading"|"ready"|"unloading","kind":"llm"|"tts","option":...,"ms":...}
  {"type":"start"}                       # engines ready, generation begins now (client resets its clock)
  {"type":"text","delta":...}
  {"type":"sentence","index":n,"text":...}
  {"type":"audio_meta","index":n,"sample_rate":..., "text":..., "seconds":...} followed by one binary PCM16 frame
  {"type":"metrics", ...}
  {"type":"error","message":...}

Only one engine per kind stays loaded; switching options unloads the previous one so VRAM stays comparable.
Engines are warmed up when first loaded so the first request is not skewed by model load time.

Tunable parameters: every request may carry {"params": {"llm": {...}, "tts": {...}, "stt": {...}, "pipeline": {...}}}.
Runtime parameters are applied to the live engine; a change to a reload-time parameter (model, device, ...)
rebuilds the engine (status events show it). GET /schema/{kind}/{option} describes the parameters for the UI,
POST /preset saves the current values as a new YAML preset.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import Body, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from llmcomm.core.params import PIPELINE_PARAMS, apply_params, coerce_overrides, effective_params, option_label, split_overrides
from llmcomm.core.prompts import DEFAULT_PROMPT, build_messages, describe_prompt, list_prompts, load_prompt, prompt_from_dict, save_prompt, style_metrics
from llmcomm.core.rag import RAG_DATA, Retriever, chunk_corpus, describe_rag, list_corpora, list_embedding_models, list_rag, load_rag, rag_from_dict, save_rag
from llmcomm.core.registry import build, describe, list_configs, save_preset, schema
from llmcomm.core.types import Message
from llmcomm.pipeline.streaming import converse

app = FastAPI(title="llmcomm")
_loaded: dict[str, tuple[str, object, dict]] = {}  # kind -> (option name, engine, reload-time overrides it was built with)
_retrievers: dict[str, Retriever] = {}  # rag option -> indexed retriever (small; several may stay loaded)


async def _retriever(name: str, ws: WebSocket, override: dict | None = None) -> Retriever:
    """Indexed retriever for a rag option; `override` (UI-edited settings) yields a separate, re-indexed instance."""
    cfg = rag_from_dict(name, {**load_rag(name).__dict__, **(override or {})}) if override else load_rag(name)
    key = name if not override else name + "~" + hashlib.sha1(json.dumps(override, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
    if key not in _retrievers:
        await ws.send_json({"type": "status", "stage": "loading", "kind": "rag", "option": name})
        t = time.perf_counter()
        r = Retriever(cfg)
        await r.index()
        await r.retrieve("워밍업")  # first embedding call loads the Ollama embedding model
        _retrievers[key] = r
        await ws.send_json({"type": "status", "stage": "ready", "kind": "rag", "option": name, "ms": round((time.perf_counter() - t) * 1000),
                            "chunks": len(r.chunks)})
    return _retrievers[key]
STATIC = Path(__file__).parent / "static"
MIC_DIR = Path(__file__).resolve().parents[3] / "data" / "stt" / "audio" / "mic"


async def _get(kind: str, name: str, ws: WebSocket, overrides: dict | None = None):
    """Return the engine for (kind, name) with `overrides` applied. Rebuild only when the option or a
    reload-time parameter changed; runtime parameters are applied in place."""
    ov = coerce_overrides(schema(kind, name), overrides or {})
    ctor_ov, live_ov = split_overrides(schema(kind, name), ov)
    cur = _loaded.get(kind)
    if cur and cur[0] == name and cur[2] == ctor_ov:
        # restore preset runtime values first so overrides from a previous request do not linger
        apply_params(cur[1], {**getattr(cur[1], "preset_params", {}), **live_ov})
        cur[1].overrides = ov
        cur[1].label = option_label(name, ov)
        return cur[1]
    if cur:
        await ws.send_json({"type": "status", "stage": "unloading", "kind": kind, "option": cur[0]})
        await cur[1].close()
        _loaded.pop(kind, None)
    await ws.send_json({"type": "status", "stage": "loading", "kind": kind, "option": name, "reload_params": ctor_ov})
    t = time.perf_counter()
    engine = build(kind, name, ov)
    await engine.warmup()
    _loaded[kind] = (name, engine, ctor_ov)
    await ws.send_json({"type": "status", "stage": "ready", "kind": kind, "option": name,
                        "ms": round((time.perf_counter() - t) * 1000)})
    return engine


def _save_mic(samples: np.ndarray, sr: int, text: str, stt_option: str) -> None:
    MIC_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    sf.write(MIC_DIR / f"{stamp}.wav", samples, sr)
    with (MIC_DIR.parent.parent / "mic_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": f"mic/{stamp}", "audio": f"audio/mic/{stamp}.wav", "text": text, "stt": stt_option,
                            "voice": "mic", "verified": False}, ensure_ascii=False) + "\n")


@app.get("/options")
def options():
    return {**{k: list_configs(k) for k in ("llm", "tts", "stt")}, "prompt": list_prompts(), "rag": ["none"] + list_rag()}


@app.get("/rag/{name}")
def rag_endpoint(name: str):
    return describe_rag(name)


@app.get("/rag-meta")
async def rag_meta():
    """Choices for the RAG editor: corpora on disk and embedding models pulled in Ollama."""
    return {"corpora": list_corpora(), "embedding_models": await list_embedding_models()}


# ---------------------------------------------------------------- RAG corpus management (upload documents from the page)

def _corpus_dir(name: str, create: bool = False) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise HTTPException(400, "corpus name: letters, digits, _ and - only")
    d = RAG_DATA / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    elif not d.exists():
        raise HTTPException(404, f"corpus {name} not found")
    return d


def _invalidate_corpus(name: str) -> None:
    """Drop cached retrievers over this corpus so the next request re-chunks and re-embeds."""
    for k in [k for k, r in _retrievers.items() if r.cfg.corpus == name]:
        _retrievers.pop(k, None)


def _to_text(filename: str, data: bytes) -> tuple[str, str]:
    """Return (stored filename, text). PDF -> .md via pypdf; .md/.txt kept; other types rejected."""
    ext = Path(filename).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9가-힣_\-]+", "_", Path(filename).stem).strip("_") or "doc"
    if ext == ".pdf":
        from io import BytesIO

        from pypdf import PdfReader

        pages = [pg.extract_text() or "" for pg in PdfReader(BytesIO(data)).pages]
        text = f"# {stem}\n\n" + "\n\n".join(f"## {stem} p.{i + 1}\n{t.strip()}" for i, t in enumerate(pages) if t.strip())
        return f"{stem}.md", text
    if ext in (".md", ".txt", ".markdown"):
        text = data.decode("utf-8-sig", errors="replace")
        if "## " not in text:  # no sections: make the file one section so chunk sources stay readable
            text = f"## {stem}\n{text}"
        return f"{stem}{'.txt' if ext == '.txt' else '.md'}", text
    raise HTTPException(400, f"unsupported file type {ext}; use .md, .txt or .pdf")


@app.get("/corpus/{name}")
def corpus_files(name: str):
    d = _corpus_dir(name)
    files = sorted(list(d.glob("*.md")) + list(d.glob("*.txt")))
    chunks = chunk_corpus(d, 400, 1)
    per_file = {}
    for c in chunks:
        per_file[c.source.split("#")[0]] = per_file.get(c.source.split("#")[0], 0) + 1
    return {"corpus": name, "files": [{"name": f.name, "bytes": f.stat().st_size, "chunks": per_file.get(f.name, 0)} for f in files],
            "chunks": len(chunks)}


@app.post("/corpus/{name}")
def corpus_create(name: str):
    _corpus_dir(name, create=True)
    return {"corpora": list_corpora()}


@app.post("/corpus/{name}/upload")
async def corpus_upload(name: str, files: list[UploadFile] = File(...)):
    d = _corpus_dir(name, create=True)
    saved = []
    for f in files:
        fname, text = _to_text(f.filename or "doc.md", await f.read())
        (d / fname).write_text(text, encoding="utf-8")
        saved.append(fname)
    _invalidate_corpus(name)
    return {"saved": saved, **corpus_files(name), "corpora": list_corpora()}


@app.delete("/corpus/{name}/file/{filename}")
def corpus_delete_file(name: str, filename: str):
    d = _corpus_dir(name)
    target = d / Path(filename).name
    if not target.exists() or target.suffix.lower() not in (".md", ".txt"):
        raise HTTPException(404, "file not found")
    target.unlink()
    _invalidate_corpus(name)
    return corpus_files(name)


@app.post("/prompt/save")
def prompt_save(body: dict = Body(...)):
    """{"name", "config": {description, persona, style, few_shot, extra}} -> configs/prompt/<name>.yaml"""
    path = save_prompt(body["name"], body["config"])
    return {"written": str(path), "options": list_prompts()}


@app.post("/rag/save")
def rag_save(body: dict = Body(...)):
    """{"name", "config": {...RagConfig fields}} -> configs/rag/<name>.yaml"""
    path = save_rag(body["name"], body["config"])
    return {"written": str(path), "options": ["none"] + list_rag()}


@app.get("/prompt/{name}")
def prompt_endpoint(name: str):
    """Persona, style rules and the composed system text of a prompt option."""
    return describe_prompt(name)


@app.get("/loaded")
def loaded():
    return {k: v[0] for k, v in _loaded.items()}


@app.get("/schema/{kind}/{option}")
def schema_endpoint(kind: str, option: str):
    """Tunable parameters + preset values for the option; `pipeline` returns the converse() knobs."""
    if kind == "pipeline":
        return {"kind": "pipeline", "option": "pipeline", "params": [dict(p.to_dict(), preset=p.default, value=p.default) for p in PIPELINE_PARAMS]}
    return describe(kind, option)


@app.post("/preset")
def preset_endpoint(body: dict = Body(...)):
    """{"kind","base","name","overrides"} -> writes configs/<kind>/<name>.yaml."""
    path = save_preset(body["kind"], body["name"], body["base"], body.get("overrides") or {})
    return {"written": str(path), "options": list_configs(body["kind"])}


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            req = await ws.receive_json()
            try:
                prompt_name = req.get("prompt") or DEFAULT_PROMPT
                if req.get("prompt_config"):  # edited in the UI, not saved: compose from the dict, label with '*'
                    pcfg = prompt_from_dict(prompt_name, req["prompt_config"])
                    prompt_name += "*"
                else:
                    pcfg = load_prompt(prompt_name)
                history = [Message(m["role"], m["content"]) for m in req.get("messages", [])]
                rag_name = req.get("rag") if req.get("rag") not in (None, "", "none") else None
                rag_override = req.get("rag_config") or None
                params = req.get("params") or {}
                pipeline = coerce_overrides(PIPELINE_PARAMS, params.get("pipeline") or {})
                if req.get("type") == "audio_in":
                    pcm = await ws.receive_bytes()
                    stt = await _get("stt", req["stt"], ws, params.get("stt"))
                    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    sr = int(req.get("sample_rate", 16000))
                    t = time.perf_counter()
                    text = await stt.transcribe(samples, sr)
                    stt_ms = round((time.perf_counter() - t) * 1000)
                    _save_mic(samples, sr, text, req["stt"])
                    await ws.send_json({"type": "transcript", "text": text, "ms": stt_ms, "seconds": round(len(samples) / sr, 2)})
                    if not text.strip():
                        await ws.send_json({"type": "error", "message": "음성에서 텍스트를 인식하지 못했습니다."})
                        continue
                    history.append(Message("user", text))
                # retrieval on the latest user utterance, then the system prompt gets the context block
                context, rag_info = None, None
                if rag_name and history and history[-1].role == "user":
                    retriever = await _retriever(rag_name, ws, rag_override)
                    hits, timing = await retriever.retrieve(history[-1].content)
                    context = retriever.context_block(hits)
                    rag_info = {"option": rag_name + ("*" if rag_override else ""), **timing, "hits": [{"source": h.source, "score": round(h.score, 3), "text": h.text} for h in hits]}
                    await ws.send_json({"type": "retrieval", **rag_info})
                # client history carries no system message; the prompt option (or a free-text override) supplies it
                msgs = build_messages(pcfg, history, system_override=req.get("system_override"), context=context)
                reply_parts: list[str] = []
                llm = await _get("llm", req["llm"], ws, params.get("llm"))
                tts = await _get("tts", req["tts"], ws, params.get("tts"))
                if req.get("stt"):
                    await _get("stt", req["stt"], ws, params.get("stt"))
                if req.get("warmup_only"):
                    await ws.send_json({"type": "metrics", "warmup_only": True})
                    continue
                n_sent = n_audio = 0
                await ws.send_json({"type": "start", "llm": llm.label, "tts": tts.label, "pipeline": pipeline, "prompt": prompt_name,
                                    "system": msgs[0].content, "rag": (rag_name + "*") if rag_name and rag_override else rag_name,
                                    "prompt_chars": sum(len(m.content) for m in msgs)})
                async for ev, p in converse(llm, tts, msgs, **pipeline, **req.get("gen", {})):
                    if ev == "text":
                        reply_parts.append(p)
                        await ws.send_json({"type": "text", "delta": p})
                    elif ev == "sentence":
                        await ws.send_json({"type": "sentence", "index": n_sent, "text": p})
                        n_sent += 1
                    elif ev == "audio":
                        pcm = (np.clip(p.samples, -1, 1) * 32767).astype(np.int16).tobytes()
                        await ws.send_json({"type": "audio_meta", "index": n_audio, "sample_rate": p.sample_rate,
                                            "text": p.text, "seconds": len(p.samples) / p.sample_rate})
                        await ws.send_bytes(pcm)
                        n_audio += 1
                    elif ev == "metrics":
                        await ws.send_json({"type": "metrics", **p, "style": style_metrics("".join(reply_parts), pcfg.style),
                                            "retrieval_ms": (rag_info or {}).get("retrieval_ms"), "context_chars": (rag_info or {}).get("context_chars"),
                                            "effective": {
                            "llm": {"option": llm.label, **effective_params(llm)},
                            "tts": {"option": tts.label, **effective_params(tts)},
                            "pipeline": pipeline}})
            except WebSocketDisconnect:
                raise
            except Exception as e:  # report to the page instead of killing the socket
                await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
    except WebSocketDisconnect:
        pass


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")
