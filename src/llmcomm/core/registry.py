"""Config-driven factory. A config YAML has `type: <registered name>` plus kwargs (a *preset*).

`build(kind, name, overrides)` layers tunable-parameter overrides on the preset: reload-time parameters go
into the constructor, runtime ones are applied to the live engine. See core/params.py.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import yaml

from .params import Param, apply_params, coerce_overrides, effective_params, option_label, schema_of, split_overrides

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
    "faster_whisper": "llmcomm.stt.faster_whisper:FasterWhisperSTT",  # GPU (or CPU int8); size name or converted CT2 dir
    "sherpa_onnx": "llmcomm.stt.sherpa_onnx:SherpaOnnxSTT",  # CPU ONNX: SenseVoice, Korean zipformer (offline/streaming)
    "vosk": "llmcomm.stt.vosk:VoskSTT",  # CPU floor baseline
}
TABLES = {"llm": LLM_TYPES, "tts": TTS_TYPES, "stt": STT_TYPES}


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


def engine_class(kind: str, name: str):
    cfg = load_config(kind, name)
    table = TABLES[kind]
    if cfg["type"] not in table:
        raise KeyError(f"Unknown {kind} type '{cfg['type']}'. Known: {sorted(table)}")
    return _load_class(table[cfg["type"]])


def schema(kind: str, name: str) -> list[Param]:
    """Tunable parameters of the adapter behind a config."""
    return schema_of(engine_class(kind, name))


def build(kind: str, name: str, overrides: dict[str, Any] | None = None):
    cls = engine_class(kind, name)
    cfg = load_config(kind, name)
    cfg.pop("type")
    cfg_name = cfg.pop("_name")
    params = schema_of(cls)
    ov = coerce_overrides(params, overrides or {})
    ctor_ov, live_ov = split_overrides(params, ov)
    # Ollama keeps sampling knobs under `options`; a preset may declare them there too.
    if "options" in cfg and isinstance(cfg["options"], dict):
        preset_opts = cfg["options"]
    else:
        preset_opts = None
    for k, v in ctor_ov.items():
        cfg[k] = v
    inst = cls(**cfg)
    if preset_opts is not None and hasattr(inst, "options"):
        inst.options = dict(preset_opts)
    # runtime values as the preset defines them, so a later request without overrides can restore them
    inst.preset_params = {k: v for k, v in effective_params(inst).items() if not any(p.name == k and p.reload for p in params)}
    apply_params(inst, live_ov)
    inst.name = cfg_name
    inst.label = option_label(cfg_name, ov)
    inst.overrides = ov
    return inst


def describe(kind: str, name: str, overrides: dict[str, Any] | None = None) -> dict:
    """Schema + preset values for the UI, without instantiating heavy models.

    Preset values come from the YAML (including Ollama `options`), falling back to schema defaults.
    """
    cls = engine_class(kind, name)
    cfg = load_config(kind, name)
    flat = {k: v for k, v in cfg.items() if k not in ("type", "_name", "options")}
    flat.update(cfg.get("options") or {})
    ov = coerce_overrides(schema_of(cls), overrides or {})
    out = []
    for p in schema_of(cls):
        d = p.to_dict()
        d["preset"] = flat.get(p.name, p.default)
        d["value"] = ov.get(p.name, d["preset"])
        out.append(d)
    return {"kind": kind, "option": name, "type": cfg["type"], "params": out, "label": option_label(name, ov)}


def current_params(engine) -> dict[str, Any]:
    return effective_params(engine)


def list_configs(kind: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_ROOT / kind).glob("*.yaml"))


def save_preset(kind: str, name: str, base: str, overrides: dict[str, Any]) -> Path:
    """Write configs/<kind>/<name>.yaml = base preset + overrides (Ollama sampling knobs go under options)."""
    cfg = load_config(kind, base)
    cfg.pop("_name")
    cls = engine_class(kind, base)
    ov = coerce_overrides(schema_of(cls), overrides)
    opt_keys = getattr(cls, "_OPTION_KEYS", set())
    for k, v in ov.items():
        if k in opt_keys:
            cfg.setdefault("options", {})[k] = v
        else:
            cfg[k] = v
    path = CONFIG_ROOT / kind / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path
