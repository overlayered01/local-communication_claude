from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

from llmcomm.core.interfaces import TTSEngine
from llmcomm.core.types import AudioChunk

from ._util import run_blocking, to_mono_float32


class WindowsSAPITTS(TTSEngine):
    """Windows built-in voices via PowerShell System.Speech. Zero install; quality floor baseline."""

    def __init__(self, voice: str | None = None, rate: int = 0):
        self.voice, self.rate = voice, rate

    def _synth(self, text: str, voice: str | None):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.wav"
            sel = f"$s.SelectVoice('{voice}');" if voice else ""
            script = (
                "[Console]::InputEncoding=[System.Text.Encoding]::UTF8;"
                "Add-Type -AssemblyName System.Speech;"
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                f"{sel}$s.Rate={self.rate};"
                f"$s.SetOutputToWaveFile('{out}');"
                "$s.Speak([Console]::In.ReadToEnd());$s.Dispose()"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", script], input=text.encode("utf-8"), check=True)
            samples, sr = sf.read(out, dtype="float32", always_2d=True)
        return to_mono_float32(samples.mean(axis=1)), sr

    async def synthesize(self, text: str, voice: str | None = None) -> AudioChunk:
        samples, sr = await run_blocking(self._synth, text, voice or self.voice)
        self.sample_rate = sr
        return AudioChunk(samples=samples, sample_rate=sr, text=text)
