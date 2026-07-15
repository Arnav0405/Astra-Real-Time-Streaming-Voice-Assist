"""Gate metrics on LibriParty: frame AUC on eval split, max-F1 threshold from dev.

Writes runs/vad/eval_report.json (threshold feeds export).

Usage:
    uv run python -m astra_ml.evaluation.eval --config configs/vad_v1.yaml \
        --checkpoint runs/vad/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from astra_ml.models.vad import VadModel
from astra_ml.training.config import load_config
from astra_ml.training.train import make_loader, pick_device


@torch.no_grad()
def collect(model: VadModel, loader, device) -> tuple[np.ndarray, np.ndarray]:
    probs, labels = [], []
    for pcm, target in loader:
        probs.append(model(pcm.to(device)).cpu().flatten().numpy())
        labels.append(target.flatten().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def max_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vad_v1.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    model = VadModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dev_probs, dev_labels = collect(model, make_loader(cfg, "dev", shuffle=False), device)
    threshold = max_f1_threshold(dev_labels, dev_probs)

    eval_probs, eval_labels = collect(model, make_loader(cfg, "eval", shuffle=False), device)
    preds = eval_probs >= threshold
    report = {
        "eval_auc": float(roc_auc_score(eval_labels, eval_probs)),
        "threshold": threshold,
        "eval_f1": float(f1_score(eval_labels, preds)),
        "eval_precision": float(precision_score(eval_labels, preds)),
        "eval_recall": float(recall_score(eval_labels, preds)),
        "gate_auc_0.95": bool(roc_auc_score(eval_labels, eval_probs) >= 0.95),
    }
    out = cfg.training.runs_dir / "eval_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report → {out}")


if __name__ == "__main__":
    main()
