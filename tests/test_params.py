import pytest

from llmcomm.core.params import Param, coerce_overrides, option_label, parse_set, parse_sweep, split_overrides
from llmcomm.core.registry import build, describe, list_configs, schema


def test_parse_set_and_sweep():
    assert parse_set(["temperature=0.3", "num_ctx = 4096"]) == {"temperature": "0.3", "num_ctx": "4096"}
    assert parse_sweep("beam_size=1,3,5") == ("beam_size", ["1", "3", "5"])
    assert parse_sweep(None) is None
    with pytest.raises(ValueError):
        parse_set(["oops"])


def test_coerce_and_split():
    params = [Param("temperature", "float", 0.7), Param("think", "bool", False), Param("model", "str", "", reload=True),
              Param("voice", "choice", "F1", choices=["F1", "M1"])]
    ov = coerce_overrides(params, {"temperature": "0.3", "think": "true", "model": "x", "voice": "M1"})
    assert ov == {"temperature": 0.3, "think": True, "model": "x", "voice": "M1"}
    ctor, live = split_overrides(params, ov)
    assert ctor == {"model": "x"} and set(live) == {"temperature", "think", "voice"}
    with pytest.raises(KeyError):
        coerce_overrides(params, {"nope": 1})
    with pytest.raises(ValueError):
        coerce_overrides(params, {"voice": "Z9"})


def test_option_label_is_stable_and_sorted():
    assert option_label("a", None) == "a"
    assert option_label("a", {"z": 1, "b": 0.5, "t": True}) == "a{b=0.5,t=true,z=1}"


def test_every_config_has_schema_and_describe_works():
    for kind in ("llm", "tts", "stt"):
        for name in list_configs(kind):
            params = schema(kind, name)
            assert params, f"{kind}/{name} declares no PARAMS"
            d = describe(kind, name)
            assert d["params"] and all("value" in p for p in d["params"])


def test_ollama_runtime_override_lands_in_options_without_network():
    eng = build("llm", "ollama_qwen3_8b", {"temperature": "0.2", "num_ctx": "4096", "think": "false"})
    assert eng.options["temperature"] == 0.2 and eng.options["num_ctx"] == 4096
    assert eng.think is False
    assert eng.label == "ollama_qwen3_8b{num_ctx=4096,temperature=0.2,think=false}"
    assert eng.get_param("top_p") == 0.9  # schema default when the preset does not set it


def test_supertonic_runtime_override_sets_attributes():
    eng = build("tts", "supertonic_gpu", {"total_steps": 4, "voice": "M1"})
    assert eng.total_steps == 4 and eng.voice == "M1" and eng.device == "cuda"
