"""Segment-level eval: score post-processed VAD events, not raw frame probs.

Complements eval.py (frame-level, raw model quality). This measures what the
user experiences after the postproc state machine: missed utterances, false
alarms per hour, speech-onset latency. Postproc params come from the model
sidecar JSON — the same file the Go runtime reads.

Usage:
    uv run python -m astra_ml.evaluation.segment_eval --config configs/vad_v1.yaml \
        --checkpoint runs/vad/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from astra_ml.audio.dft_mel import FRAME_SAMPLES
from astra_ml.models.vad import VadModel
from astra_ml.postproc import PostprocConfig, segments
from astra_ml.training.config import load_config
from astra_ml.training.train import make_loader, pick_device

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SIDECAR = REPO_ROOT / "assets" / "models" / "vad" / "vad_v1.json"
FRAME_MS = FRAME_SAMPLES / 16_000 * 1000


def label_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of 1s in a binary label array → (start, end) frame pairs."""
    padded = np.diff(np.concatenate([[0], labels.astype(int), [0]]))
    starts = np.flatnonzero(padded == 1)
    ends = np.flatnonzero(padded == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def segment_metrics(true_segs: list[tuple[int, int]], pred_segs: list[tuple[int, int]]) -> dict:
    """Recall over true segments, false-alarm count over preds, onset latency per match.

    A true segment is matched if any prediction overlaps it; latency is measured
    against the first overlapping prediction (negative = started early).
    """
    latencies = []
    matched = 0
    for t in true_segs:
        hit = next((p for p in pred_segs if _overlaps(t, p)), None)
        if hit is not None:
            matched += 1
            latencies.append(hit[0] - t[0])
    false_alarms = sum(1 for p in pred_segs if not any(_overlaps(t, p) for t in true_segs))
    return {
        "recall": matched / len(true_segs) if true_segs else None,
        "matched": matched,
        "false_alarms": false_alarms,
        "onset_latencies": latencies,
    }


def score_windows(windows, cfg: PostprocConfig) -> dict:
    """Aggregate segment metrics over (probs, labels) window pairs."""
    matched = total_true = false_alarms = total_frames = 0
    latencies: list[int] = []
    for window_probs, window_labels in windows:
        true_segs = label_segments(window_labels)
        m = segment_metrics(true_segs, segments(window_probs, cfg))
        total_true += len(true_segs)
        matched += m["matched"]
        false_alarms += m["false_alarms"]
        latencies += m["onset_latencies"]
        total_frames += len(window_probs)
    hours = total_frames * FRAME_MS / 1000 / 3600
    lat_ms = np.array(latencies) * FRAME_MS
    return {
        "segment_recall": matched / total_true if total_true else None,
        "false_alarms_per_hour": false_alarms / hours,
        "median_onset_latency_ms": float(np.median(lat_ms)) if latencies else None,
        "p90_onset_latency_ms": float(np.percentile(lat_ms, 90)) if latencies else None,
        "true_segments": total_true,
        "audio_hours": hours,
    }


@torch.no_grad()
def collect_windows(model: VadModel, loader, device) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run inference once, return per-window (probs, labels) pairs."""
    windows = []
    for pcm, target in loader:
        probs = model(pcm.to(device)).cpu().numpy()
        windows += list(zip(probs, target.numpy(), strict=True))
    return windows


def evaluate(model: VadModel, loader, device, cfg: PostprocConfig) -> dict:
    return score_windows(collect_windows(model, loader, device), cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vad_v1.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pp_cfg = PostprocConfig.from_sidecar(json.loads(args.sidecar.read_text()))
    device = pick_device()
    model = VadModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    report = evaluate(model, make_loader(cfg, "eval", shuffle=False), device, pp_cfg)
    report["postproc"] = pp_cfg.__dict__
    out = cfg.training.runs_dir / "segment_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report → {out}")


if __name__ == "__main__":
    main()
