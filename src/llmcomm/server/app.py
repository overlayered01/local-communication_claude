"""FastAPI server: WebSocket /ws/chat streams text + PCM audio for any llm/tts option pair.

Client sends JSON: {"llm": "...", "tts": "...", "messages": [{"role": ..., "content": ...}]}
Server sends:
  {"type":"text","delta":...}
  {"type":"sentence","text":...}
  {"type":"audio_meta","sample_rate":..., "text":...} followed by one binary PCM16 frame
  {"type":"metrics", ...}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from llmcomm.core.registry import build, list_configs
from llmcomm.core.types import Message
from llmcomm.pipeline.streaming import converse

app = FastAPI(title="llmcomm")
_cache: dict[tuple[str, str], object] = {}
STATIC = Path(__file__).parent / "static"


def _get(kind: str, name: str):
    key = (kind, name)
    if key not in _cache:
        _cache[key] = build(kind, name)
    return _cache[key]


@app.get("/options")
def options():
    return {k: list_configs(k) for k in ("llm", "tts", "stt")}


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    while True:
        req = await ws.receive_json()
        llm, tts = _get("llm", req["llm"]), _get("tts", req["tts"])
        msgs = [Message(m["role"], m["content"]) for m in req["messages"]]
        async for ev, p in converse(llm, tts, msgs):
            if ev == "text":
                await ws.send_json({"type": "text", "delta": p})
            elif ev == "sentence":
                await ws.send_json({"type": "sentence", "text": p})
            elif ev == "audio":
                await ws.send_json({"type": "audio_meta", "sample_rate": p.sample_rate, "text": p.text})
                await ws.send_bytes((np.clip(p.samples, -1, 1) * 32767).astype(np.int16).tobytes())
            elif ev == "metrics":
                await ws.send_json({"type": "metrics", **p})


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")
