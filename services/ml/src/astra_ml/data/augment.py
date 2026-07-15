"""CHiME-Home augmentation: standalone non-speech negatives + runtime noise mixing.

Both consume the dev_nonspeech.csv manifest written by astra_ml.data.chime
(rows: chunk_name, wav_path — same shape as the eval manifests).
"""

import csv
import random
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from astra_ml.audio.dft_mel import FRAME_SAMPLES


def load_manifest(path: Path) -> list[Path]:
    with open(path) as f:
        return [Path(row[1]) for row in csv.reader(f)]


def _read_crop(wav: Path, n_samples: int, offset: int = 0) -> torch.Tensor:
    pcm, _ = sf.read(wav, dtype="float32", always_2d=True)
    pcm = torch.from_numpy(pcm[:, 0])
    if len(pcm) < offset + n_samples:
        reps = -(-(offset + n_samples) // len(pcm))
        pcm = pcm.repeat(reps)
    return pcm[offset : offset + n_samples]


class ChimeNegatives(Dataset):
    """CHiME non-speech chunks as all-zero-label crops. Deterministic per seed.

    reverse=True draws from the opposite end of the shuffled pool so train and
    dev negatives stay disjoint while the pool lasts; cycles via modulo beyond it.
    """

    def __init__(
        self, manifest: Path, crop_frames: int, count: int, seed: int, reverse: bool = False
    ) -> None:
        self.crop_frames = crop_frames
        paths = load_manifest(manifest)
        random.Random(seed).shuffle(paths)
        if reverse:
            paths = paths[::-1]
        self.items = [paths[i % len(paths)] for i in range(count)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pcm = _read_crop(self.items[idx], self.crop_frames * FRAME_SAMPLES)
        return pcm, torch.zeros(self.crop_frames)


class NoiseAugment(Dataset):
    """Wraps a dataset; with prob `prob` adds a CHiME chunk at random SNR. Labels unchanged."""

    def __init__(
        self, base: Dataset, manifest: Path, prob: float, snr_db_range: tuple[float, float]
    ) -> None:
        self.base = base
        self.noises = load_manifest(manifest)
        self.prob = prob
        self.snr_lo, self.snr_hi = snr_db_range

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pcm, labels = self.base[idx]
        if torch.rand(()) >= self.prob:
            return pcm, labels
        noise = _read_crop(self.noises[torch.randint(len(self.noises), ()).item()], len(pcm))
        noise_power = noise.square().mean()
        if noise_power == 0:
            return pcm, labels
        snr_db = self.snr_lo + (self.snr_hi - self.snr_lo) * torch.rand(())
        gain = torch.sqrt(pcm.square().mean() / (noise_power * 10 ** (snr_db / 10)))
        pcm = pcm + gain * noise
        peak = pcm.abs().max()
        if peak > 1:
            pcm = pcm / peak
        return pcm, labels
