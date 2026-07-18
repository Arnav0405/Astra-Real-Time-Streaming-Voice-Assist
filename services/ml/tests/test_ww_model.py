"""Tests for the wake-word head model."""

import torch

from astra_ml.models.ww import EMB_DIM, HEAD_FRAMES, WwHead


def test_shapes():
    model = WwHead(layer_size=16)
    feats = torch.zeros(5, HEAD_FRAMES, EMB_DIM)
    assert model.logits(feats).shape == (5,)
    probs = model(feats)
    assert probs.shape == (5,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_overfits_tiny_synthetic_set():
    torch.manual_seed(0)
    feats = torch.randn(32, HEAD_FRAMES, EMB_DIM)
    labels = (torch.arange(32) < 16).float()
    feats[labels > 0.5] += 1.0  # separable shift

    model = WwHead(layer_size=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss()
    for _ in range(200):
        opt.zero_grad()
        loss = criterion(model.logits(feats), labels)
        loss.backward()
        opt.step()

    acc = ((model(feats) >= 0.5).float() == labels).float().mean().item()
    assert acc == 1.0
