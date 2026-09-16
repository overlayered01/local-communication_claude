from llmcomm.bench.judge import _parse
from llmcomm.bench.runner import DEFAULT_SYSTEM, _build_messages


def test_judge_parse_plain_json():
    assert _parse('{"naturalness": 4, "korean": 5, "relevance": 3, "brevity": 4}') == {
        "naturalness": 4.0, "korean": 5.0, "relevance": 3.0, "brevity": 4.0}


def test_judge_parse_strips_think_and_prose():
    raw = "<think>생각 중...</think>평가 결과입니다:\n{\"naturalness\": 2, \"korean\": 3, \"relevance\": 4, \"brevity\": 5}\n끝."
    assert _parse(raw)["brevity"] == 5.0


def test_judge_parse_invalid_returns_none():
    assert _parse("점수를 매길 수 없습니다.") is None
    assert _parse('{"naturalness": 4}') is None


def test_build_messages_single_turn():
    msgs = _build_messages({"id": "x", "user": "안녕"})
    assert [m.role for m in msgs] == ["system", "user"]
    assert msgs[0].content == DEFAULT_SYSTEM


def test_build_messages_multiturn_and_system_override():
    p = {"id": "x", "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]}
    msgs = _build_messages(p, system="S")
    assert msgs[0].content == "S"
    assert [m.content for m in msgs[1:]] == ["a", "b", "c"]
