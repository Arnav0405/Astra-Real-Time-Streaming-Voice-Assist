"""Dataset-free smoke test for the wake-word v2 trainer."""

import torch
from test_kws_dataset import _cfg, corpus  # noqa: F401  (fixture reuse)

from astra_ml.training.train_kws import SpecAugment, load_model, lr_factor, train


def test_lr_factor_warms_up_then_anneals_to_zero():
    warmup, total = 100, 1000
    assert lr_factor(0, warmup, total) == 0.0
    assert lr_factor(50, warmup, total) == 0.5
    assert lr_factor(warmup, warmup, total) == 1.0
    assert lr_factor(total, warmup, total) < 1e-9
    mid = lr_factor((warmup + total) // 2, warmup, total)
    assert 0.4 < mid < 0.6


def test_specaugment_masks_whole_frequency_rows():
    mel = torch.ones(4, 1, 40, 94)
    aug = SpecAugment(freq_param=5, time_param=0, n_freq=2, n_time=0).train()
    out = aug(mel)
    zeroed_rows = (out == 0).all(dim=-1)
    assert zeroed_rows.any(), "no frequency row was masked"
    assert zeroed_rows.sum().item() <= 4 * 2 * 5  # at most n_freq * freq_param rows per item


def test_specaugment_masks_whole_time_columns():
    mel = torch.ones(4, 1, 40, 94)
    aug = SpecAugment(freq_param=0, time_param=20, n_freq=0, n_time=2).train()
    out = aug(mel)
    assert (out == 0).all(dim=-2).any(), "no time column was masked"


def test_specaugment_is_identity_in_eval():
    mel = torch.randn(2, 1, 40, 94)
    aug = SpecAugment(5, 20, 2, 2).eval()
    torch.testing.assert_close(aug(mel), mel)


def test_train_smoke_produces_a_loadable_checkpoint(corpus):  # noqa: F811
    cfg = _cfg(
        corpus,
        steps=6,
        val_every=3,
        val_batches=1,
        warmup_steps=2,
        num_workers=0,
        seed=7,
    )
    best = train(cfg)
    assert best.exists()

    model = load_model(best)
    assert model.training is False
    with torch.no_grad():
        assert model(torch.randn(1, 1, cfg.frontend.n_mels, 94)).shape == (1, 1)
