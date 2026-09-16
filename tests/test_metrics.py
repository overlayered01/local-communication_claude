from llmcomm.bench.metrics import cer


def test_cer():
    assert cer("안녕하세요", "안녕하세요") == 0.0
    assert cer("안녕하세요.", "안녕 하세요") == 0.0  # spacing/punct ignored
    assert 0 < cer("안녕하세요", "안녕하셔요") < 0.5
