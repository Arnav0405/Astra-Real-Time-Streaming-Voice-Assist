"""Plain-PyTorch VAD training loop.

Usage (GPU machine, after data.generate):
    uv run python -m astra_ml.training.train --config configs/vad_v1.yaml
Monitor: uv run tensorboard --logdir runs/vad
"""

import argparse
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from astra_ml.data.libriparty import LibriPartyDataset
from astra_ml.models.vad import VadModel
from astra_ml.training.config import Config, load_config


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(cfg: Config, split: str, shuffle: bool) -> DataLoader:
    dataset = LibriPartyDataset(
        cfg.data.metadata_json(split),
        cfg.data.mixtures_dir(split),
        crop_frames=cfg.training.crop_frames,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate_auc(model: VadModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    probs, labels = [], []
    for pcm, target in loader:
        probs.append(model(pcm.to(device)).cpu().flatten())
        labels.append(target.flatten())
    return roc_auc_score(torch.cat(labels).numpy(), torch.cat(probs).numpy())


def train(config_path: Path) -> Path:
    cfg = load_config(config_path)
    t = cfg.training
    torch.manual_seed(t.seed)
    device = pick_device()
    print(f"device: {device}")

    train_loader = make_loader(cfg, "train", shuffle=True)
    dev_loader = make_loader(cfg, "dev", shuffle=False)
    print(f"train windows: {len(train_loader.dataset)}, dev windows: {len(dev_loader.dataset)}")

    model = VadModel().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=t.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t.epochs)

    t.runs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(t.runs_dir)
    best_path = t.runs_dir / "best.pt"
    best_auc, stale = 0.0, 0

    for epoch in range(t.epochs):
        model.train()
        total, batches = 0.0, 0
        for pcm, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model.logits(pcm.to(device)), target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
        scheduler.step()

        dev_auc = evaluate_auc(model, dev_loader, device)
        writer.add_scalar("loss/train", total / max(batches, 1), epoch)
        writer.add_scalar("auc/dev", dev_auc, epoch)
        print(f"epoch {epoch:3d}  loss {total / max(batches, 1):.4f}  dev AUC {dev_auc:.4f}")

        if dev_auc > best_auc:
            best_auc, stale = dev_auc, 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= t.patience:
                print(f"early stop at epoch {epoch} (best dev AUC {best_auc:.4f})")
                break

    writer.close()
    print(f"best dev AUC {best_auc:.4f} → {best_path}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vad_v1.yaml"))
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
