from __future__ import annotations

from llmcomm.core.interfaces import TTSEngine
from llmcomm.core.types import AudioChunk
from llmcomm.core.params import Param

from ._util import decode_audio_bytes


class EdgeTTS(TTSEngine):
    """Microsoft Edge neural voices (cloud). NOT local: used only as a quality/latency reference point."""


    PARAMS = [
        Param("voice", "choice", "ko-KR-SunHiNeural", choices=["ko-KR-SunHiNeural", "ko-KR-InJoonNeural", "ko-KR-HyunsuMultilingualNeural"], description="Edge 음성"),
        Param("rate", "str", "+0%", description="속도 (예: +10%, -5%)"),
    ]

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
