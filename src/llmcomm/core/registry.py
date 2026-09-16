"""Config-driven factory. A config YAML has `type: <registered name>` plus kwargs."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"

# type name -> "module:Class". Modules are imported lazily so optional deps only load when used.
LLM_TYPES: dict[str, str] = {
    "ollama": "llmcomm.llm.ollama:OllamaBackend",
    "openai_compat": "llmcomm.llm.openai_compat:OpenAICompatBackend",  # vLLM, llama.cpp server, LM Studio
    "llamacpp": "llmcomm.llm.llamacpp:LlamaCppBackend",
}
TTS_TYPES: dict[str, str] = {
    "supertonic": "llmcomm.tts.supertonic:SupertonicTTS",
    "melotts": "llmcomm.tts.melotts:MeloTTS",
    "cosyvoice": "llmcomm.tts.cosyvoice:CosyVoiceTTS",
    "edge": "llmcomm.tts.edge:EdgeTTS",  # cloud baseline for quality comparison only
    "sapi": "llmcomm.tts.sapi:WindowsSAPITTS",  # zero-install floor baseline
}
STT_TYPES: dict[str, str] = {
    "faster_whisper": "llmcomm.stt.faster_whisper:FasterWhisperSTT",
}


def _load_class(spec: str) -> Callable[..., Any]:
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def load_config(kind: str, name: str) -> dict:
    path = CONFIG_ROOT / kind / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (CONFIG_ROOT / kind).glob("*.yaml"))
        raise FileNotFoundError(f"{path} not found. Available {kind}: {available}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg["_name"] = name
    return cfg


def build(kind: str, name: str):
    table = {"llm": LLM_TYPES, "tts": TTS_TYPES, "stt": STT_TYPES}[kind]
    cfg = load_config(kind, name)
    type_name = cfg.pop("type")
    cfg_name = cfg.pop("_name")
    if type_name not in table:
        raise KeyError(f"Unknown {kind} type '{type_name}'. Known: {sorted(table)}")
    inst = _load_class(table[type_name])(**cfg)
    inst.name = cfg_name
    return inst


def list_configs(kind: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_ROOT / kind).glob("*.yaml"))
