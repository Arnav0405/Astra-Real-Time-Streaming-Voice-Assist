"""Dataset-free tests for feature precompute and the training loop."""

import csv

import numpy as np
import pytest
import soundfile as sf
import torch
from test_ww_frontend import fake_embed, fake_melspec

from astra_ml.audio.ww_frontend import DEFAULT_FRONTEND, WwFrontend
from astra_ml.models.ww import EMB_DIM, HEAD_FRAMES
from astra_ml.training.train_ww import (
    FeaturePools,
    _slide_features,
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


def _write_negative_recordings(cfg, seconds=3.0):
    root = cfg.data.negative_recordings_root
    (root / "session_neg_a").mkdir(parents=True)
    sf.write(
        root / "session_neg_a" / "000.wav",
        np.random.default_rng(2).normal(0, 0.1, int(SR * seconds)).astype(np.float32),
        SR,
    )
    with open(root / "manifest_split.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "session", "speaker", "env", "split"])
        writer.writerow(["session_neg_a/000.wav", "neg_a", "user", "", "train"])
        # an eval-split row must stay out of the training pool: it is the speech FA set
        writer.writerow(["session_neg_a/000.wav", "neg_a", "user", "", "eval"])


def test_slide_features_covers_every_hop():
    fe = WwFrontend(fake_melspec, fake_embed, DEFAULT_FRONTEND)
    ws = fe.cfg.window_samples
    hop = int(0.08 * SR)
    clip = np.zeros(ws + 5 * hop, dtype=np.float32)
    # every 80 ms offset the runtime can score, not just the 3 positive jitters
    assert len(_slide_features(fe, clip)) == 6


def test_slide_features_pads_a_short_clip_to_one_window():
    fe = WwFrontend(fake_melspec, fake_embed, DEFAULT_FRONTEND)
    feats = _slide_features(fe, np.zeros(SR // 4, dtype=np.float32))
    assert len(feats) == 1
    assert feats[0].shape == (HEAD_FRAMES, EMB_DIM)


def test_precompute_caches_personal_negatives(ww_cfg):
    _write_tts_tree(ww_cfg)
    _write_recordings(ww_cfg)
    _write_negative_recordings(ww_cfg)
    fe = WwFrontend(fake_melspec, fake_embed, DEFAULT_FRONTEND)
    cache = precompute(ww_cfg, frontend=fe)

    user_neg = np.load(cache / "negatives_user.npy")
    # 1 train clip * 2 augment rounds, slid; speed augmentation makes the exact window
    # count vary, so pin the shape and that both rounds landed rather than a magic number
    assert user_neg.shape[1:] == (HEAD_FRAMES, EMB_DIM)
    assert user_neg.dtype == np.float16
    assert len(user_neg) >= 2 * ww_cfg.augment.augment_rounds_user_negative


def test_precompute_without_personal_negatives_caches_an_empty_pool(ww_cfg):
    _write_tts_tree(ww_cfg)
    _write_recordings(ww_cfg)
    fe = WwFrontend(fake_melspec, fake_embed, DEFAULT_FRONTEND)
    cache = precompute(ww_cfg, frontend=fe)
    assert np.load(cache / "negatives_user.npy").shape == (0, HEAD_FRAMES, EMB_DIM)


def test_precompute_shapes(ww_cfg):
    _write_tts_tree(ww_cfg)
    _write_recordings(ww_cfg)
    fe = WwFrontend(fake_melspec, fake_embed, DEFAULT_FRONTEND)
    cache = precompute(ww_cfg, frontend=fe)

    pos = np.load(cache / "positives.npy")
    # 2 TTS train clips * 3 jitters; user clips are cached separately
    assert pos.shape == (6, HEAD_FRAMES, EMB_DIM)
    assert pos.dtype == np.float16
    # 1 user clip * 2 augment rounds * 3 jitters
    assert np.load(cache / "positives_user.npy").shape == (6, HEAD_FRAMES, EMB_DIM)
    assert np.load(cache / "positives_val.npy").shape == (3, HEAD_FRAMES, EMB_DIM)
    assert np.load(cache / "negatives_adv.npy").shape == (6, HEAD_FRAMES, EMB_DIM)
    assert np.load(cache / "negatives_local.npy").shape == (0, HEAD_FRAMES, EMB_DIM)


def _write_feature_cache(cfg, n_pos=8, n_val=4, n_adv=8, n_user=8, n_user_neg=0):
    cache = cfg.training.features_cache
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    shape = (HEAD_FRAMES, EMB_DIM)
    np.save(cache / "positives.npy", rng.normal(1, 0.1, (n_pos, *shape)).astype(np.float16))
    # marked at 5.0 so a batch can be traced back to which positive pool it came from
    np.save(cache / "positives_user.npy", np.full((n_user, *shape), 5.0, dtype=np.float16))
    np.save(cache / "positives_val.npy", rng.normal(1, 0.1, (n_val, *shape)).astype(np.float16))
    np.save(cache / "negatives_adv.npy", rng.normal(0, 0.1, (n_adv, *shape)).astype(np.float16))
    np.save(cache / "negatives_local.npy", np.zeros((0, *shape), dtype=np.float16))
    # 7.0 for the same reason 5.0 marks user positives
    np.save(cache / "negatives_user.npy", np.full((n_user_neg, *shape), 7.0, dtype=np.float16))


# No ACAV file in the fixture, so its share of every batch is drawn from a zero-length
# pool and the negative half comes up short of n_neg. These assert shares, not totals.
def test_personal_negatives_get_a_quarter_of_the_negative_batch(ww_cfg):
    _write_feature_cache(ww_cfg, n_user_neg=200)
    pools = FeaturePools(ww_cfg, seed=0)
    feats, labels = pools.batch(400)
    neg = feats[labels < 0.5]
    from_user = (neg.reshape(len(neg), -1)[:, 0] == 7.0).sum().item()
    n_neg = 400 - 400 // 4
    # equal share with adv and local: the only pool covering the deployment channel
    assert from_user == n_neg // 4
    assert len(neg) - from_user == n_neg // 4  # the adv share, local/acav pools being empty


def test_missing_personal_negatives_does_not_starve_the_batch(ww_cfg):
    _write_feature_cache(ww_cfg, n_user_neg=0)
    pools = FeaturePools(ww_cfg, seed=0)
    feats, labels = pools.batch(400)
    neg = feats[labels < 0.5]
    assert len(neg) == (400 - 400 // 4) // 4  # adv only, exactly as before this pool existed
    assert (neg.reshape(len(neg), -1)[:, 0] == 7.0).sum().item() == 0
    assert labels.sum() == 400 // 4  # positives unaffected


def test_feature_pools_batch_composition(ww_cfg):
    _write_feature_cache(ww_cfg)
    pools = FeaturePools(ww_cfg, seed=0)  # no ACAV file -> zero-length pool
    feats, labels = pools.batch(8)
    assert feats.shape[1:] == (HEAD_FRAMES, EMB_DIM)
    assert labels[: 8 // 4].sum() == 2  # first quarter positive
    assert labels[8 // 4 :].sum() == 0


def test_user_positive_frac_controls_the_mix(ww_cfg):
    _write_feature_cache(ww_cfg)
    ww_cfg.training.user_positive_frac = 0.5
    pools = FeaturePools(ww_cfg, seed=0)
    feats, labels = pools.batch(400)
    pos = feats[labels > 0.5]
    from_user = (pos.reshape(len(pos), -1)[:, 0] == 5.0).sum().item()
    assert len(pos) == 100  # a quarter of the batch, as before
    assert from_user == 50  # half of those drawn from real recordings, not TTS


def test_user_positive_frac_zero_falls_back_to_tts_only(ww_cfg):
    _write_feature_cache(ww_cfg, n_user=0)
    ww_cfg.training.user_positive_frac = 0.5
    pools = FeaturePools(ww_cfg, seed=0)  # empty user pool must not starve the batch
    feats, labels = pools.batch(400)
    pos = feats[labels > 0.5]
    assert len(pos) == 100
    assert (pos.reshape(len(pos), -1)[:, 0] == 5.0).sum().item() == 0


def test_train_writes_loadable_checkpoint(ww_cfg, capsys):
    _write_feature_cache(ww_cfg)
    best = train(ww_cfg)
    assert best.exists()
    model = load_head(best)
    out = model(torch.zeros(2, HEAD_FRAMES, EMB_DIM))
    assert out.shape == (2,)


def test_config_rejects_overlapping_folds(tmp_path):
    # a fold in two roles silently restores the leak the split exists to remove
    from astra_ml.training.ww_config import WwDataConfig, WwEvalConfig, _check_folds_disjoint

    data = WwDataConfig(
        frontends_dir=tmp_path, voices_dir=tmp_path, voices=[], tts_out=tmp_path,
        spellings=[], n_positives=0, n_positives_val=0, adversarial_phrases=[],
        n_adversarial_per_phrase=0, acav_features=tmp_path, acav_subsample=0,
        negative_audio_dirs=[], recordings_root=tmp_path,
        negative_folds=[1, 2, 3], negative_val_folds=[4],
    )
    ok = WwEvalConfig([], 0.95, 0.8, 0.5, 500.0, fa_folds=[5])
    _check_folds_disjoint(data, ok)  # disjoint: fine

    leaky = WwEvalConfig([], 0.95, 0.8, 0.5, 500.0, fa_folds=[3, 5])
    with pytest.raises(ValueError, match="negative_folds and fa_folds share fold"):
        _check_folds_disjoint(data, leaky)


def test_precompute_writes_held_out_local_val_pool(ww_cfg, tmp_path):
    neg = tmp_path / "neg"
    neg.mkdir()
    # two ESC-50-named clips, one in a train fold and one in the val fold
    for fold in (1, 4):
        sf.write(neg / f"{fold}-1000-A-0.wav", np.zeros(int(SR * 3.0), dtype=np.float32), SR)
    ww_cfg.data.negative_audio_dirs = [neg]
    ww_cfg.data.negative_folds = [1]
    ww_cfg.data.negative_val_folds = [4]
    _write_tts_tree(ww_cfg)
    _write_recordings(ww_cfg)

    cache = precompute(ww_cfg, frontend=WwFrontend(fake_melspec, fake_embed, DEFAULT_FRONTEND))
    train_neg = np.load(cache / "negatives_local.npy")
    val_neg = np.load(cache / "negatives_local_val.npy")
    # both folds yield windows, and the val pool exists so est_fa_per_hour has a domain to score
    assert len(train_neg) > 0
    assert len(val_neg) > 0
    assert train_neg.shape[1:] == val_neg.shape[1:] == (HEAD_FRAMES, EMB_DIM)


def test_feature_pools_rejects_corrupt_acav(ww_cfg):
    # a killed curl leaves a partial file that exists() passes; fail loudly, not mid-train
    _write_feature_cache(ww_cfg)
    ww_cfg.data.acav_features.write_bytes(b"not an npy file")
    with pytest.raises(ValueError, match="ACAV"):
        FeaturePools(ww_cfg, seed=0)
