"""Dataset-free tests for feature precompute and the training loop."""

import csv

import numpy as np
import soundfile as sf
import torch
from test_ww_frontend import fake_embed, fake_melspec

from astra_ml.audio.ww_frontend import DEFAULT_FRONTEND, WwFrontend
from astra_ml.models.ww import EMB_DIM, HEAD_FRAMES
from astra_ml.training.train_ww import (
    FeaturePools,
    clip_features,
    load_head,
    place_clip,
    precompute,
    train,
)

SR = 16000


def test_place_clip_end_alignment():
    clip = np.ones(1000, dtype=np.float32)
    win = place_clip(clip, end_offset=0, window_samples=4000)
    assert win[3000:].sum() == 1000
    assert win[:3000].sum() == 0
    win = place_clip(clip, end_offset=500, window_samples=4000)
    assert win[2500:3500].sum() == 1000
    assert win[3500:].sum() == 0


def test_place_clip_crops_long_clip_from_start():
    clip = np.arange(6000, dtype=np.float32)
    win = place_clip(clip, end_offset=0, window_samples=4000)
    assert win[0] == 2000  # keeps the tail of the clip
    assert win[-1] == 5999


def test_clip_features_jitter_count():
    fe = WwFrontend(fake_melspec, fake_embed, DEFAULT_FRONTEND)
    feats = clip_features(fe, np.zeros(SR // 2, dtype=np.float32))
    assert feats.shape == (3, HEAD_FRAMES, EMB_DIM)


def _write_tts_tree(cfg):
    rows = []
    rng = np.random.default_rng(0)

    def wav(rel, seconds=0.4):
        path = cfg.data.tts_out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, rng.normal(0, 0.1, int(SR * seconds)).astype(np.float32), SR)
        return rel

    for i in range(2):
        rows.append((wav(f"positives/{i}.wav"), "positive", "astraa", "train"))
    rows.append((wav("positives/2.wav"), "positive", "astraa", "val"))
    for i in range(2):
        rows.append((wav(f"adversarial/astro/{i}.wav"), "adversarial", "astro", "train"))
    with open(cfg.data.tts_out / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "phrase", "split"])
        writer.writerows(rows)


def _write_recordings(cfg):
    root = cfg.data.recordings_root
    (root / "session_a").mkdir(parents=True)
    sf.write(
        root / "session_a" / "000.wav",
        np.random.default_rng(1).normal(0, 0.1, SR // 2).astype(np.float32),
        SR,
    )
    with open(root / "manifest_split.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "session", "speaker", "env", "split"])
        writer.writerow(["session_a/000.wav", "a", "user", "", "train"])


def test_precompute_shapes(ww_cfg):
    _write_tts_tree(ww_cfg)
    _write_recordings(ww_cfg)
    fe = WwFrontend(fake_melspec, fake_embed, DEFAULT_FRONTEND)
    cache = precompute(ww_cfg, frontend=fe)

    pos = np.load(cache / "positives.npy")
    # 2 TTS train clips * 3 jitters + 1 user clip * 2 augment rounds * 3 jitters
    assert pos.shape == (12, HEAD_FRAMES, EMB_DIM)
    assert pos.dtype == np.float16
    assert np.load(cache / "positives_val.npy").shape == (3, HEAD_FRAMES, EMB_DIM)
    assert np.load(cache / "negatives_adv.npy").shape == (6, HEAD_FRAMES, EMB_DIM)
    assert np.load(cache / "negatives_local.npy").shape == (0, HEAD_FRAMES, EMB_DIM)


def _write_feature_cache(cfg, n_pos=8, n_val=4, n_adv=8):
    cache = cfg.training.features_cache
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    shape = (HEAD_FRAMES, EMB_DIM)
    np.save(cache / "positives.npy", rng.normal(1, 0.1, (n_pos, *shape)).astype(np.float16))
    np.save(cache / "positives_val.npy", rng.normal(1, 0.1, (n_val, *shape)).astype(np.float16))
    np.save(cache / "negatives_adv.npy", rng.normal(0, 0.1, (n_adv, *shape)).astype(np.float16))
    np.save(cache / "negatives_local.npy", np.zeros((0, *shape), dtype=np.float16))


def test_feature_pools_batch_composition(ww_cfg):
    _write_feature_cache(ww_cfg)
    pools = FeaturePools(ww_cfg, seed=0)  # no ACAV file -> zero-length pool
    feats, labels = pools.batch(8)
    assert feats.shape[1:] == (HEAD_FRAMES, EMB_DIM)
    assert labels[: 8 // 4].sum() == 2  # first quarter positive
    assert labels[8 // 4 :].sum() == 0


def test_train_writes_loadable_checkpoint(ww_cfg, capsys):
    _write_feature_cache(ww_cfg)
    best = train(ww_cfg)
    assert best.exists()
    model = load_head(best)
    out = model(torch.zeros(2, HEAD_FRAMES, EMB_DIM))
    assert out.shape == (2,)
