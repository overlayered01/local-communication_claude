import numpy as np

from llmcomm.bench.metrics import cer
from llmcomm.core.registry import STT_TYPES, list_configs, load_config
from llmcomm.stt._util import pick, to_16k


def test_to_16k_resamples_and_flattens():
    x = np.zeros((44100, 2), dtype=np.float32)
    y = to_16k(x, 44100)
    assert y.ndim == 1 and y.dtype == np.float32
    assert abs(len(y) - 16000) <= 1
    assert to_16k(np.zeros(16000, dtype=np.float32), 16000).shape == (16000,)


def test_pick_prefers_requested_quantization(tmp_path):
    for n in ["encoder-epoch-99-avg-1.onnx", "encoder-epoch-99-avg-1.int8.onnx"]:
        (tmp_path / n).write_bytes(b"")
    assert pick(tmp_path, "encoder*.onnx", int8=True).endswith("int8.onnx")
    assert not pick(tmp_path, "encoder*.onnx", int8=False).endswith("int8.onnx")
    (tmp_path / "only.int8.onnx").write_bytes(b"")
    assert pick(tmp_path, "only*.onnx", int8=False).endswith("only.int8.onnx")  # falls back to what exists


def test_every_stt_config_has_registered_type():
    for name in list_configs("stt"):
        cfg = load_config("stt", name)
        assert cfg["type"] in STT_TYPES, f"{name}: unknown type {cfg['type']}"


def test_cer_ignores_spacing_and_punctuation():
    assert cer("오늘은 날씨가 정말 좋네요.", "오늘은날씨가정말좋네요") == 0.0
    assert 0 < cer("주말에 뭐 하면 좋을지", "주말에 뭐 하면 좋을까") < 0.2
