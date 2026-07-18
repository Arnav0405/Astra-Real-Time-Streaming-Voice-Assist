"""Dataset-free tests for the wake-word feature frontend."""

from pathlib import Path

import numpy as np
import pytest

from astra_ml.audio.ww_frontend import DEFAULT_FRONTEND, FrontendConfig, WwFrontend

FRONTENDS_DIR = Path(__file__).parents[1] / "datasets" / "oww_frontends"


def fake_melspec(x: np.ndarray) -> np.ndarray:
    """Deterministic stand-in honoring frames(n) = (n - 640) // 160 + 1.

    Frame f encodes (frame index, mean of its samples) so window slicing is checkable.
    """
    cfg = DEFAULT_FRONTEND
    n = x.shape[1]
    frames = (n - cfg.mel_window) // cfg.mel_hop + 1
    out = np.zeros((frames, cfg.mel_bins), dtype=np.float32)
    for f in range(frames):
        out[f, 0] = f
        out[f, 1] = x[0, f * cfg.mel_hop : f * cfg.mel_hop + cfg.mel_window].mean()
    return out


def fake_embed(windows: np.ndarray) -> np.ndarray:
    """Embedding dim 0 = first mel frame index of the window (post-transform)."""
    batch = windows.shape[0]
    out = np.zeros((batch, 96), dtype=np.float32)
    out[:, 0] = windows[:, 0, 0, 0]
    out[:, 1] = windows[:, :, 1, 0].mean(axis=1)
    return out


def test_config_pinned_arithmetic():
    cfg = DEFAULT_FRONTEND
    assert cfg.window_frames_total == 196
    assert cfg.window_samples == 31840
    streamed = (cfg.chunk_samples + cfg.mel_lookback - cfg.mel_window) // cfg.mel_hop + 1
    assert cfg.mel_frames_per_chunk == streamed


def test_features_shape_and_window_slicing():
    cfg = DEFAULT_FRONTEND
    fe = WwFrontend(fake_melspec, fake_embed, cfg)
    audio = np.zeros(cfg.window_samples, dtype=np.float32)
    feats = fe.features(audio)
    assert feats.shape == (cfg.head_frames, cfg.emb_dim)
    # window i starts at mel frame i*stride; fake encodes transformed frame index
    for i in range(cfg.head_frames):
        expected = (i * cfg.emb_stride) * cfg.transform_scale + cfg.transform_offset
        assert feats[i, 0] == pytest.approx(expected)


def test_features_rejects_wrong_length():
    fe = WwFrontend(fake_melspec, fake_embed)
    with pytest.raises(ValueError):
        fe.features(np.zeros(1000, dtype=np.float32))


def test_melspec_transform_applied():
    fe = WwFrontend(fake_melspec, fake_embed)
    audio = np.full(DEFAULT_FRONTEND.window_samples, 100.0, dtype=np.float32)
    spec = fe.melspec(audio)
    # fake mel bin 1 = mean of raw samples = 100 -> transformed 100*0.1 + 2
    assert spec[0, 1] == pytest.approx(12.0)


@pytest.mark.skipif(
    not (FRONTENDS_DIR / "melspectrogram.onnx").exists(),
    reason="frozen OWW frontends not downloaded",
)
def test_parity_with_openwakeword_audiofeatures():
    from openwakeword.utils import AudioFeatures

    cfg = DEFAULT_FRONTEND
    rng = np.random.default_rng(0)
    audio = (rng.uniform(-0.5, 0.5, cfg.window_samples) * 32767).astype(np.int16)

    af = AudioFeatures(
        melspec_model_path=str(FRONTENDS_DIR / "melspectrogram.onnx"),
        embedding_model_path=str(FRONTENDS_DIR / "embedding_model.onnx"),
        inference_framework="onnx",
    )
    want = np.asarray(af._get_embeddings(audio)).reshape(cfg.head_frames, cfg.emb_dim)

    fe = WwFrontend.from_onnx(FRONTENDS_DIR, cfg)
    got = fe.features(audio.astype(np.float32))
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_nondefault_config_scales():
    cfg = FrontendConfig(head_frames=4)
    assert cfg.window_frames_total == 76 + 3 * 8
    fe = WwFrontend(fake_melspec, fake_embed, cfg)
    feats = fe.features(np.zeros(cfg.window_samples, dtype=np.float32))
    assert feats.shape == (4, 96)
