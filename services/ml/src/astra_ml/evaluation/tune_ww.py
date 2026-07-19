"""Grid-search the wake-word trigger knobs against the dev split.

Score streams are computed once (the expensive part), then every
threshold × patience combo is re-scored through the cheap postproc machine.
Objective mirrors tune_postproc: lowest false-accepts/hour subject to the
recall floors (quiet and noisy, on the "eval" split — never the frozen test
sessions), tie-break lower median latency.

    uv run python -m astra_ml.evaluation.tune_ww --config configs/ww_v1.yaml \
        --checkpoint runs/ww/best.pt [--update-sidecar]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from astra_ml.audio.ww_frontend import DEFAULT_FRONTEND, WwFrontend
from astra_ml.evaluation.ww_eval import (
    SR,
    collect_streams,
    fa_per_hour,
    LEAD_IN_FRAMES,
    recall_and_latency,
)
from astra_ml.postproc_ww import WwPostprocConfig
from astra_ml.training.train_ww import load_head
from astra_ml.training.ww_config import load_ww_config

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SIDECAR = REPO_ROOT / "assets" / "models" / "wakeword" / "ww_v1.json"

THRESHOLDS = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
PATIENCES = [1, 2, 3]


def score_grid(collected: dict, refractory_frames: int) -> list[dict]:
    results = []
    for threshold in THRESHOLDS:
        for patience in PATIENCES:
            pp_cfg = WwPostprocConfig(threshold, patience, refractory_frames)
            recall_q, lat_q, _ = recall_and_latency(
                collected["recall"]["quiet"],
                collected["word_end"]["quiet"],
                pp_cfg,
                min_frame=LEAD_IN_FRAMES,
            )
            recall_n, lat_n, _ = recall_and_latency(
                collected["recall"]["noisy"],
                collected["word_end"]["noisy"],
                pp_cfg,
                min_frame=LEAD_IN_FRAMES,
            )
            latencies = lat_q + lat_n
            results.append(
                {
                    "threshold": threshold,
                    "patience_frames": patience,
                    "recall_quiet": recall_q,
                    "recall_noisy": recall_n,
                    "fa_per_hour": fa_per_hour(collected["fa"], pp_cfg),
                    "latency_ms_median": float(np.median(latencies)) if latencies else None,
                }
            )
    return results


def pick_best(results: list[dict], floor_quiet: float, floor_noisy: float) -> dict:
    ok = [
        r for r in results if r["recall_quiet"] >= floor_quiet and r["recall_noisy"] >= floor_noisy
    ]
    if not ok:
        # nothing meets the floors: maximize recall instead so the report is useful
        return max(results, key=lambda r: r["recall_quiet"] + r["recall_noisy"])
    return min(
        ok,
        key=lambda r: (
            r["fa_per_hour"],
            r["latency_ms_median"] if r["latency_ms_median"] is not None else float("inf"),
        ),
    )


def update_sidecar(sidecar_path: Path, best: dict) -> None:
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["recommended_threshold"] = best["threshold"]
    sidecar["postproc"]["patience_frames"] = best["patience_frames"]
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v1.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/ww/best.pt"))
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--update-sidecar", action="store_true")
    args = parser.parse_args()
    cfg = load_ww_config(args.config)

    frontend = WwFrontend.from_onnx(cfg.data.frontends_dir, DEFAULT_FRONTEND)
    head = load_head(args.checkpoint)
    refractory_frames = int(cfg.postproc.refractory_seconds * SR / 320)

    collected = collect_streams(cfg, frontend, head)
    results = score_grid(collected, refractory_frames)
    best = pick_best(results, cfg.eval.recall_floor_quiet, cfg.eval.recall_floor_noisy)

    report = {
        "best": {**best, "refractory_frames": refractory_frames},
        "floors": {
            "recall_quiet": cfg.eval.recall_floor_quiet,
            "recall_noisy": cfg.eval.recall_floor_noisy,
        },
        "combos_scored": len(results),
        "results": sorted(results, key=lambda r: r["fa_per_hour"])[:10],
    }
    out = cfg.training.runs_dir / "tune_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["best"], indent=2))
    print(f"report: {out}")

    if args.update_sidecar:
        update_sidecar(args.sidecar, best)
        print(f"sidecar updated: {args.sidecar}")


if __name__ == "__main__":
    main()
