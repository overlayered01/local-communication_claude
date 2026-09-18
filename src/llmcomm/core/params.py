"""Tunable parameters for engines.

Every adapter declares `PARAMS: list[Param]`. A YAML config is a *preset* of those parameters; callers
(CLI `--set`, web tester form) layer *overrides* on top. Two kinds of parameter:

  reload=False  runtime knob read on every call (temperature, speed, beam_size...). Applied to a live
                engine with `apply_params()` -> `engine.set_param()`; no model reload.
  reload=True   construction-time setting (model size, device, compute type). Changing it needs a rebuild.

Results carry the effective parameter set (`effective_params`) and a label (`option_label`) such as
`ollama_qwen3_8b{temperature=0.3,num_ctx=4096}` so tuned runs stay separate rows in the report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ParamType = Literal["float", "int", "bool", "str", "choice"]


@dataclass
class Param:
    name: str
    type: ParamType
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    reload: bool = False
    description: str = ""
    group: str = ""  # UI grouping hint

    def coerce(self, value: Any) -> Any:
        if value is None:
            return None
        if self.type == "float":
            return float(value)
        if self.type == "int":
            return int(float(value))
        if self.type == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on", "y")
            return bool(value)
        if self.type == "choice" and self.choices and str(value) not in self.choices:
            raise ValueError(f"{self.name}: {value!r} not in {self.choices}")
        return str(value) if self.type in ("str", "choice") else value

    def to_dict(self) -> dict:
        return asdict(self)


# Parameters of the streaming pipeline itself (not tied to one engine).
PIPELINE_PARAMS: list[Param] = [
    Param("tts_concurrency", "int", 2, 1, 4, 1, description="동시에 합성하는 문장 수", group="pipeline"),
    Param("min_chars", "int", 6, 1, 30, 1, description="이보다 짧은 문장은 다음 문장과 합쳐 TTS로 보냄", group="pipeline"),
]


def schema_of(obj_or_cls) -> list[Param]:
    return list(getattr(obj_or_cls, "PARAMS", []))


def param_map(params: list[Param]) -> dict[str, Param]:
    return {p.name: p for p in params}


def parse_set(pairs: list[str] | None) -> dict[str, str]:
    """`--set k=v --set k2=v2` -> {k: v}. Values are strings; coerced against the schema later."""
    out: dict[str, str] = {}
    for kv in pairs or []:
        if "=" not in kv:
            raise ValueError(f"--set expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_sweep(spec: str | None) -> tuple[str, list[str]] | None:
    """`--sweep temperature=0.3,0.7,1.0` -> ("temperature", ["0.3","0.7","1.0"])."""
    if not spec:
        return None
    k, vs = spec.split("=", 1)
    return k.strip(), [v.strip() for v in vs.split(",") if v.strip()]


def coerce_overrides(params: list[Param], overrides: dict[str, Any]) -> dict[str, Any]:
    pm = param_map(params)
    out: dict[str, Any] = {}
    for k, v in overrides.items():
        if k not in pm:
            raise KeyError(f"unknown parameter {k!r}; available: {sorted(pm)}")
        out[k] = pm[k].coerce(v)
    return out


def split_overrides(params: list[Param], overrides: dict[str, Any]) -> tuple[dict, dict]:
    """(reload-time overrides, runtime overrides)."""
    pm = param_map(params)
    ctor = {k: v for k, v in overrides.items() if pm[k].reload}
    live = {k: v for k, v in overrides.items() if not pm[k].reload}
    return ctor, live


def option_label(name: str, overrides: dict[str, Any] | None) -> str:
    if not overrides:
        return name
    inner = ",".join(f"{k}={_fmt(v)}" for k, v in sorted(overrides.items()))
    return f"{name}{{{inner}}}"


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def apply_params(engine, overrides: dict[str, Any]) -> None:
    """Apply runtime overrides to a live engine. Adapters may override `set_param` for nested storage."""
    for k, v in overrides.items():
        setter = getattr(engine, "set_param", None)
        if setter:
            setter(k, v)
        else:
            setattr(engine, k, v)


def effective_params(engine) -> dict[str, Any]:
    """Current value of every declared parameter, read back from the engine."""
    out: dict[str, Any] = {}
    for p in schema_of(engine):
        getter = getattr(engine, "get_param", None)
        out[p.name] = getter(p.name) if getter else getattr(engine, p.name, p.default)
    return out


@dataclass
class ParamState:
    """What the server keeps per loaded engine to decide whether a change needs a rebuild."""
    option: str
    ctor_overrides: dict[str, Any] = field(default_factory=dict)
