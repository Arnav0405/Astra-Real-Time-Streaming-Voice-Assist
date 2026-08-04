"""Train wake word v2: BC-ResNet on log-mel, from scratch.

    uv run python -m astra_ml.training.train_kws --config configs/ww_v2.yaml

No precompute step, unlike train_ww. The frontend is part of the model now, so windows are
built and augmented in DataLoader workers, and the log-mel plus SpecAugment run on the GPU
with the batch. Everything the paper specifies is here: SGD with momentum, weight decay
1e-3, linear warmup to lr 0.1 then cosine to zero, dropout 0.1 and SubSpectralNorm inside
the blocks, and SpecAugment with two time and two frequency masks (no time warping).

Checkpoint selection is inherited from train_ww: recall at a fixed FPR on the negative
validation pool, never validation loss. The gate that matters is false accepts per hour,
and a model can improve its loss while getting worse at exactly that.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from astra_ml.audio.dft_mel import LogMelSTFT
from astra_ml.data.kws_dataset import KwsWindows, build_pools
from astra_ml.models.bcresnet import BCResNets
from astra_ml.training.kws_config import KwsConfig, load_kws_config

# 80 ms scoring cadence, matching ww_eval and the Go detector. Used to turn a validation
# false-positive rate into the units the eval.max_fa_per_hour gate is written in.
SCORE_STEPS_PER_HOUR = 16000 * 3600 / 1280
SELECT_FPR = 1e-3
# Loose enough to never bind in normal training; present because SGD at lr 0.1 is a
# regime this repo has not run before (everything prior was Adam at 1.5e-3).
GRAD_CLIP_NORM = 5.0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SpecAugment(nn.Module):
    """Two frequency and two time masks, no time warping (the paper drops it).

    Mask width follows model capacity: BC-ResNet-{1.5, 2, 3, 6, 8} use F of {1, 3, 5, 7, 7}
    with T fixed at 20, and BC-ResNet-1 uses none.
    """

    def __init__(self, freq_param: int, time_param: int, n_freq: int, n_time: int) -> None:
        super().__init__()
        self.masks = nn.ModuleList(
            [torchaudio.transforms.FrequencyMasking(freq_param) for _ in range(n_freq)]
            + [torchaudio.transforms.TimeMasking(time_param) for _ in range(n_time)]
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mel
        for mask in self.masks:
            mel = mask(mel)
        return mel


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup from zero, then cosine annealing to zero."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def make_loader(cfg: KwsConfig, split: str, length: int, seed: int) -> DataLoader:
    pools = build_pools(cfg, split)
    pools.check()
    dataset = KwsWindows(cfg, pools, length=length, seed=seed)
    workers = cfg.training.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,  # composition is positional; see KwsWindows
        num_workers=workers,
        drop_last=True,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def precompute_val_mels(
    cfg: KwsConfig, frontend: LogMelSTFT, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validation windows are fixed for the whole run, so their log-mels are computed once.

    Caching mels rather than waveforms costs ~60 MB instead of ~245 MB and makes every
    validation pass a pure forward.
    """
    t = cfg.training
    loader = make_loader(cfg, "val", length=t.val_batches * t.batch_size, seed=t.seed + 1)
    mels, labels = [], []
    for window, label in loader:
        mels.append(frontend(window.to(device)).cpu())
        labels.append(label)
    return torch.cat(mels), torch.cat(labels)


@torch.no_grad()
def _scores(model: nn.Module, mels: torch.Tensor, device: torch.device) -> torch.Tensor:
    out = []
    for i in range(0, len(mels), 256):
        out.append(torch.sigmoid(model(mels[i : i + 256].to(device))).squeeze(1).cpu())
    return torch.cat(out) if out else torch.zeros(0)


def validate(
    model: nn.Module,
    mels: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    scores = _scores(model, mels, device)
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]

    recall = pos.ge(0.5).float().mean().item() if len(pos) else 0.0
    fp_rate = neg.ge(0.5).float().mean().item() if len(neg) else 0.0
    far = neg.ge(threshold).float().mean().item() if len(neg) else 0.0
    recall_pp = pos.ge(threshold).float().mean().item() if len(pos) else 0.0

    if len(neg):
        cut = torch.quantile(neg, 1.0 - SELECT_FPR).item()
        select = pos.gt(cut).float().mean().item() if len(pos) else 0.0
    else:
        select = recall

    tp = pos.ge(threshold).float().sum().item()
    fp = neg.ge(threshold).float().sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall_pp / (precision + recall_pp) if precision + recall_pp else 0.0
    auc = roc_auc_score(labels.numpy(), scores.numpy()) if len(pos) and len(neg) else 0.0
    return {
        "recall": recall,
        "fp_rate": fp_rate,
        "recall_at_select_fpr": select,
        "est_fa_per_hour": far * SCORE_STEPS_PER_HOUR,
        "far": far,
        "frr": 1.0 - recall_pp,
        "f1": f1,
        "auc": float(auc),
    }


def train(cfg: KwsConfig) -> Path:
    t, m = cfg.training, cfg.model
    torch.manual_seed(t.seed)
    np.random.seed(t.seed)
    device = pick_device()
    print(f"device: {device}")

    frontend = LogMelSTFT(
        n_samples=cfg.frontend.window_samples,
        win_samples=cfg.frontend.win_samples,
        hop_samples=cfg.frontend.hop_samples,
        n_mels=cfg.frontend.n_mels,
    ).to(device)
    specaug = SpecAugment(
        cfg.augment.freq_mask_param,
        cfg.augment.time_mask_param,
        cfg.augment.n_freq_masks,
        cfg.augment.n_time_masks,
    ).to(device)
    model = BCResNets(base_c=m.base_c, num_classes=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BC-ResNet base_c={m.base_c}: {n_params / 1000:.1f}k params, {frontend.n_frames} frames")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=t.lr_peak, momentum=t.momentum, weight_decay=t.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, t.warmup_steps, t.steps)
    )
    criterion = nn.BCEWithLogitsLoss()

    val_mels, val_labels = precompute_val_mels(cfg, frontend, device)
    print(f"val: {int(val_labels.sum())} positive / {len(val_labels)} windows")

    t.runs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(t.runs_dir)
    best_path = t.runs_dir / "best.pt"
    best_score = -1.0

    loader = make_loader(cfg, "train", length=t.steps * t.batch_size, seed=t.seed)
    for step, (window, label) in enumerate(loader, start=1):
        model.train()
        window, label = window.to(device), label.to(device)
        mel = specaug(frontend(window))
        loss = criterion(model(mel).squeeze(1), label)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()
        writer.add_scalar("loss/train", loss.item(), step)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], step)

        if step % t.val_every == 0 or step == t.steps:
            metrics = validate(model, val_mels, val_labels, device, cfg.postproc.threshold)
            for name, value in metrics.items():
                writer.add_scalar(f"val/{name}", value, step)
            print(
                f"step {step}: loss {loss.item():.4f} "
                + " ".join(f"{k} {v:.4f}" for k, v in metrics.items())
            )
            if metrics["recall_at_select_fpr"] > best_score:
                best_score = metrics["recall_at_select_fpr"]
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "base_c": m.base_c,
                        "window_samples": cfg.frontend.window_samples,
                    },
                    best_path,
                )

    writer.close()
    print(f"best: {best_path} (recall@fpr {best_score:.4f})")
    return best_path


def load_model(checkpoint: Path) -> BCResNets:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = BCResNets(base_c=ckpt["base_c"], num_classes=1)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v2.yaml"))
    args = parser.parse_args()
    train(load_kws_config(args.config))


if __name__ == "__main__":
    main()
