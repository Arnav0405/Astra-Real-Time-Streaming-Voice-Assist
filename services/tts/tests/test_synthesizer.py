"""Real-voice tests for synthesizer.Synthesizer. Skipped without the model."""

from pathlib import Path

import pytest

from synthesizer import Synthesizer

MODEL = (
    Path(__file__).resolve().parents[3]
    / "assets" / "models" / "tts" / "en_US-lessac-medium.onnx"
)

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run make tts-voice first")


def test_synthesize_returns_22050hz_whole_sample_pcm():
    synth = Synthesizer(str(MODEL))
    assert synth.sample_rate == 22050
    pcm, rate = synth.synthesize("Hello world.")
    assert rate == 22050
    assert len(pcm) > 0
    assert len(pcm) % 2 == 0  # whole s16le samples, or the client reassembles noise


def test_synthesize_rejects_blank_text():
    synth = Synthesizer(str(MODEL))
    with pytest.raises(ValueError):
        synth.synthesize("   ")


def test_length_scale_changes_duration():
    text = "Hello world, this is a longer test sentence."
    normal = Synthesizer(str(MODEL), length_scale=1.0).synthesize(text)[0]
    fast = Synthesizer(str(MODEL), length_scale=0.7).synthesize(text)[0]
    assert len(fast) < len(normal)
