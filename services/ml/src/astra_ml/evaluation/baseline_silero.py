"""Informational baseline: Silero VAD on the same LibriParty eval split.

Silero consumes 512-sample chunks (32 ms) @ 16 kHz; each 20 ms frame takes the
probability of the Silero chunk containing its center. Needs internet on first
run (torch.hub download).

Usage:
    uv run python -m astra_ml.evaluation.baseline_silero --config configs/vad_v1.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from astra_ml.audio.dft_mel import FRAME_SAMPLES
from astra_ml.training.config import load_config
from astra_ml.training.train import make_loader

SILERO_CHUNK = 512


@torch.no_grad()
def silero_frame_probs(model, pcm: torch.Tensor) -> torch.Tensor:
    """[T] pcm → [T // 320] per-20ms-frame probability."""
    model.reset_states()
    chunk_probs = [
        model(pcm[i : i + SILERO_CHUNK], 16_000).item()
        for i in range(0, len(pcm) - SILERO_CHUNK + 1, SILERO_CHUNK)
    ]
    n_frames = len(pcm) // FRAME_SAMPLES
    centers = (torch.arange(n_frames) * FRAME_SAMPLES + FRAME_SAMPLES // 2) // SILERO_CHUNK
    centers = centers.clamp(max=len(chunk_probs) - 1)
    return torch.tensor(chunk_probs)[centers]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vad_v1.yaml"))
    args = parser.parse_args()

    model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    model.eval()

    cfg = load_config(args.config)
    loader = make_loader(cfg, "eval", shuffle=False)
    probs, labels = [], []
    for pcm_batch, target in loader:
        for pcm, t in zip(pcm_batch, target, strict=True):
            probs.append(silero_frame_probs(model, pcm).numpy())
            labels.append(t.numpy())
    auc = roc_auc_score(np.concatenate(labels), np.concatenate(probs))
    print(f"Silero frame AUC on LibriParty eval: {auc:.4f}")


if __name__ == "__main__":
    main()
