"""Dataset-free tests for TTS generation (generator monkeypatched)."""

import csv
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from astra_ml.data import ww_generate


@pytest.fixture
def cfg(ww_cfg):
    return ww_cfg


def fake_generate_samples_onnx(text, output_dir, model, max_samples, **kwargs):
    """Write max_samples silence wavs at 22050 Hz, like piper voices do."""
    sr = 22050
    for i in range(max_samples):
        path = Path(output_dir) / f"{i}.wav"
        with wave.open(str(path), "wb") as w:
            w.setframerate(sr)
            w.setsampwidth(2)
            w.setnchannels(1)
            w.writeframes(np.zeros(sr // 2, dtype=np.int16).tobytes())


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(ww_generate, "generate_samples_onnx", fake_generate_samples_onnx)


def test_generate_all_manifest_and_resampling(cfg, patched):
    manifest = ww_generate.generate_all(cfg)
    with open(manifest) as f:
        rows = list(csv.DictReader(f))

    positives = [r for r in rows if r["label"] == "positive"]
    adversarial = [r for r in rows if r["label"] == "adversarial"]
    assert len(positives) == 8  # n_positives + n_positives_val
    assert len(adversarial) == 6  # 2 phrases x 3
    assert sum(r["split"] == "val" for r in positives) == 2
    assert {r["phrase"] for r in adversarial} == {"astro", "ad astra"}

    # every clip resampled to 16 kHz s16
    for r in rows:
        audio, sr = sf.read(cfg.data.tts_out / r["path"])
        assert sr == 16000
        assert len(audio) > 0


def test_val_split_is_tail_by_index(cfg, patched):
    manifest = ww_generate.generate_all(cfg)
    with open(manifest) as f:
        positives = [r for r in csv.DictReader(f) if r["label"] == "positive"]
    val_stems = {Path(r["path"]).stem for r in positives if r["split"] == "val"}
    assert val_stems == {"6", "7"}


def test_smoke_writes_per_spelling_dirs(cfg, patched, capsys):
    ww_generate.generate_smoke(cfg, 2)
    for spelling in ("astraa", "astra"):
        wavs = list((cfg.data.tts_out / "smoke" / spelling).glob("*.wav"))
        assert len(wavs) == 2
