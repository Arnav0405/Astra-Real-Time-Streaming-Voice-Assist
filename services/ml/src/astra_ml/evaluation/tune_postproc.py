"""Grid-search postproc knobs on the dev split against segment-level metrics.

Objective: lowest false-alarms/hour among combos with segment recall >= --min-recall,
ties broken by p90 onset latency. Inference runs once and is cached to .npz;
re-runs sweep the grid without touching the model.

Usage:
    uv run python -m astra_ml.evaluation.tune_postproc --config configs/vad_v1.yaml \
        --checkpoint runs/vad/best.pt [--update-sidecar]
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from astra_ml.evaluation.segment_eval import DEFAULT_SIDECAR, collect_windows, score_windows
from astra_ml.models.vad import VadModel
from astra_ml.postproc import PostprocConfig
from astra_ml.training.config import load_config
from astra_ml.training.train import make_loader, pick_device

ONSETS = [0.65, 0.677, 0.7, 0.77, 0.8, 0.85]
OFFSETS = [0.09, 0.1, 0.125, 0.15, 0.2]
MIN_SPEECH = [2, 4, 6, 8]
MIN_SILENCE = [32, 34, 36, 40]


def make_grid(onsets, offsets, min_speech, min_silence) -> list[PostprocConfig]:
    return [
        PostprocConfig(on, off, sp, si)
        for on, off, sp, si in itertools.product(onsets, offsets, min_speech, min_silence)
        if off < on
    ]


def pick_best(results: list[tuple], min_recall: float) -> tuple:
    """results: (key, metrics) pairs. Lowest FA/hr among recall qualifiers,
    p90 latency tiebreak; if nothing qualifies, highest recall."""
    inf = float("inf")
    qualifiers = [r for r in results if (r[1]["segment_recall"] or 0) >= min_recall]
    if not qualifiers:
        return max(results, key=lambda r: r[1]["segment_recall"] or 0)
    return min(
        qualifiers,
        key=lambda r: (
            r[1]["false_alarms_per_hour"],
            r[1]["p90_onset_latency_ms"] if r[1]["p90_onset_latency_ms"] is not None else inf,
        ),
    )


def load_or_collect(args, cfg) -> list[tuple[np.ndarray, np.ndarray]]:
    cache = cfg.training.runs_dir / f"tune_windows_{args.split}.npz"
    if cache.exists():
        data = np.load(cache)
        print(f"cached probs ← {cache}")
        return list(zip(data["probs"], data["labels"], strict=True))
    if args.checkpoint is None:
        raise SystemExit(f"no cache at {cache}; --checkpoint required for first run")
    device = pick_device()
    model = VadModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    windows = collect_windows(model, make_loader(cfg, args.split, shuffle=False), device)
    probs, labels = zip(*windows, strict=True)
    np.savez(cache, probs=np.stack(probs), labels=np.stack(labels))
    print(f"probs cached → {cache}")
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vad_v1.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--update-sidecar", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    windows = load_or_collect(args, cfg)
    grid = make_grid(ONSETS, OFFSETS, MIN_SPEECH, MIN_SILENCE)

    results = []
    for i, pp in enumerate(grid):
        results.append((pp, score_windows(windows, pp)))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(grid)} combos scored")

    best_pp, best_metrics = pick_best(results, args.min_recall)
    ranked = sorted(
        results,
        key=lambda r: (
            (r[1]["segment_recall"] or 0) < args.min_recall,  # qualifiers first
            r[1]["false_alarms_per_hour"],
            r[1]["p90_onset_latency_ms"] if r[1]["p90_onset_latency_ms"] is not None else 1e9,
        ),
    )

    print(
        f"\n{'onset':>6} {'offset':>6} {'speech':>6} {'silence':>7} "
        f"{'recall':>7} {'FA/hr':>8} {'p90ms':>7}"
    )
    for pp, m in ranked[: args.top]:
        print(
            f"{pp.onset_threshold:6.3f} {pp.offset_threshold:6.2f} "
            f"{pp.min_speech_frames:6d} {pp.min_silence_frames:7d} "
            f"{m['segment_recall'] or 0:7.3f} {m['false_alarms_per_hour']:8.1f} "
            f"{m['p90_onset_latency_ms'] or -1:7.0f}"
        )

    report = {
        "best": {**best_pp.__dict__, **best_metrics},
        "min_recall": args.min_recall,
        "split": args.split,
        "combos_scored": len(grid),
    }
    out = cfg.training.runs_dir / "tune_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nbest: {best_pp}\nreport → {out}")

    if args.update_sidecar:
        sidecar = json.loads(args.sidecar.read_text())
        sidecar["recommended_threshold"] = best_pp.onset_threshold
        sidecar["postproc"] = {
            "offset_threshold": best_pp.offset_threshold,
            "min_speech_frames": best_pp.min_speech_frames,
            "min_silence_frames": best_pp.min_silence_frames,
        }
        args.sidecar.write_text(json.dumps(sidecar, indent=2) + "\n")
        print(f"sidecar updated → {args.sidecar}")


if __name__ == "__main__":
    main()
