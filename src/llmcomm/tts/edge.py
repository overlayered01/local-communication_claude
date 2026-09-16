from __future__ import annotations

from llmcomm.core.interfaces import TTSEngine
from llmcomm.core.types import AudioChunk

from ._util import decode_audio_bytes


class EdgeTTS(TTSEngine):
    """Microsoft Edge neural voices (cloud). NOT local: used only as a quality/latency reference point."""

    def __init__(self, voice: str = "ko-KR-SunHiNeural", rate: str = "+0%"):
        self.voice, self.rate = voice, rate

    async def synthesize(self, text: str, voice: str | None = None) -> AudioChunk:
        try:
            import edge_tts
        except ImportError as e:
            raise ImportError("pip install 'llmcomm[edge]'") from e
        comm = edge_tts.Communicate(text, voice or self.voice, rate=self.rate)
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        samples, sr = decode_audio_bytes(bytes(buf))
        self.sample_rate = sr
        return AudioChunk(samples=samples, sample_rate=sr, text=text)
