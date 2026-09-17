"""FastAPI server: WebSocket /ws/chat streams text + PCM audio for any llm/tts option pair.

Client sends JSON: {"llm": "...", "tts": "...", "messages": [{"role": ..., "content": ...}]}
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
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from llmcomm.core.registry import build, list_configs
from llmcomm.core.types import Message
from llmcomm.pipeline.streaming import converse

app = FastAPI(title="llmcomm")
_loaded: dict[str, tuple[str, object]] = {}  # kind -> (option name, engine)
STATIC = Path(__file__).parent / "static"


async def _get(kind: str, name: str, ws: WebSocket):
    cur = _loaded.get(kind)
    if cur and cur[0] == name:
        return cur[1]
    if cur:
        await ws.send_json({"type": "status", "stage": "unloading", "kind": kind, "option": cur[0]})
        await cur[1].close()
        _loaded.pop(kind, None)
    await ws.send_json({"type": "status", "stage": "loading", "kind": kind, "option": name})
    t = time.perf_counter()
    engine = build(kind, name)
    await engine.warmup()
    _loaded[kind] = (name, engine)
    await ws.send_json({"type": "status", "stage": "ready", "kind": kind, "option": name,
                        "ms": round((time.perf_counter() - t) * 1000)})
    return engine


@app.get("/options")
def options():
    return {k: list_configs(k) for k in ("llm", "tts", "stt")}


@app.get("/loaded")
def loaded():
    return {k: v[0] for k, v in _loaded.items()}


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            req = await ws.receive_json()
            try:
                llm = await _get("llm", req["llm"], ws)
                tts = await _get("tts", req["tts"], ws)
                if req.get("warmup_only"):
                    await ws.send_json({"type": "metrics", "warmup_only": True})
                    continue
                msgs = [Message(m["role"], m["content"]) for m in req["messages"]]
                n_sent = n_audio = 0
                await ws.send_json({"type": "start"})
                async for ev, p in converse(llm, tts, msgs, **req.get("gen", {})):
                    if ev == "text":
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
                        await ws.send_json({"type": "metrics", **p})
            except WebSocketDisconnect:
                raise
            except Exception as e:  # report to the page instead of killing the socket
                await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
    except WebSocketDisconnect:
        pass


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")
