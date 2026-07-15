import csv

import numpy as np
import pytest
import soundfile as sf
import torch

from astra_ml.data.augment import ChimeNegatives, NoiseAugment, _read_crop


@pytest.fixture
def manifest(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3):
        wav = tmp_path / f"chunk_{i}.wav"
        sf.write(wav, (0.1 * rng.standard_normal(64_000)).astype(np.float32), 16_000)
        rows.append([f"chunk_{i}", str(wav)])
    path = tmp_path / "dev_nonspeech.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


def test_negatives_shapes_and_labels(manifest):
    ds = ChimeNegatives(manifest, crop_frames=200, count=5, seed=1)
    assert len(ds) == 5  # count > pool of 3 → modulo cycling
    pcm, labels = ds[0]
    assert pcm.shape == (200 * 320,) and labels.shape == (200,)
    assert labels.sum() == 0


def test_negatives_deterministic_and_disjoint(manifest):
    a = ChimeNegatives(manifest, crop_frames=200, count=2, seed=1)
    b = ChimeNegatives(manifest, crop_frames=200, count=2, seed=1)
    assert a.items == b.items
    dev = ChimeNegatives(manifest, crop_frames=200, count=1, seed=1, reverse=True)
    assert dev.items[0] != a.items[0]  # opposite ends of the shuffled pool


def test_read_crop_tiles_short_chunk(manifest):
    wav = ChimeNegatives(manifest, crop_frames=200, count=1, seed=0).items[0]
    pcm = _read_crop(wav, 250 * 320)  # 250 frames > 4 s chunk → tiled
    assert pcm.shape == (250 * 320,)
    assert torch.equal(pcm[64_000 : 64_000 + 100], pcm[:100])


class _Base(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return torch.full((64_000,), 0.5), torch.ones(200)


def test_noise_augment_prob_one(manifest):
    torch.manual_seed(0)
    ds = NoiseAugment(_Base(), manifest, prob=1.0, snr_db_range=(0.0, 20.0))
    assert len(ds) == 4
    pcm, labels = ds[0]
    base_pcm, base_labels = _Base()[0]
    assert torch.equal(labels, base_labels)
    assert not torch.equal(pcm, base_pcm)
    assert pcm.abs().max() <= 1.0


def test_noise_augment_prob_zero(manifest):
    ds = NoiseAugment(_Base(), manifest, prob=0.0, snr_db_range=(0.0, 20.0))
    pcm, _ = ds[0]
    assert torch.equal(pcm, _Base()[0][0])
