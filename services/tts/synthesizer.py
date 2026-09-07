"""Piper synthesis: text in, (pcm s16le mono, sample_rate_hz) out.

Whole-sentence synthesis: Piper renders the full sentence before the first
byte exists, so first-audio latency == this call's latency. Chunking into
gRPC frames happens in server.py, not here.
"""

from __future__ import annotations

from piper import PiperVoice, SynthesisConfig


class Synthesizer:
    """Thin wrapper over PiperVoice with prosody knobs and a blank-text guard."""

    def __init__(
        self,
        model_path: str,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.333,
    ):
        # config_path defaults to model_path + ".json", which is exactly
        # "<voice>.onnx.json" as shipped by the Piper release mirror.
        self.voice = PiperVoice.load(model_path)
        self._syn_config = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w,
        )

    @property
    def sample_rate(self) -> int:
        return self.voice.config.sample_rate

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """Render text to (headerless s16le mono pcm, sample_rate_hz)."""
        if not text.strip():
            raise ValueError("empty text")
        pcm = b"".join(
            chunk.audio_int16_bytes
            for chunk in self.voice.synthesize(text, syn_config=self._syn_config)
        )
        if not pcm:
            raise RuntimeError("piper produced no audio")
        return pcm, self.sample_rate
