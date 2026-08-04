"""Grid-search the wake-word trigger knobs against the dev split.

    uv run python -m astra_ml.evaluation.tune_ww --config configs/ww_v1.yaml \
        --checkpoint runs/ww/best.pt [--update-sidecar]
"""

import argparse
import json
from pathlib import Path

from astra_ml.audio.ww_frontend import DEFAULT_FRONTEND, WwFrontend
from astra_ml.evaluation.ww_eval import collect_streams
from astra_ml.evaluation.ww_metrics import (
    PATIENCES,
    SR,
    THRESHOLDS,
    pick_best,
    score_grid,
    tune_report,
    update_sidecar,
)
from astra_ml.training.train_ww import load_head
from astra_ml.training.ww_config import load_ww_config

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SIDECAR = REPO_ROOT / "assets" / "models" / "wakeword" / "ww_v1.json"

# Re-exported so importers (and tests) keep the pre-split import surface.
__all__ = [
    "PATIENCES",
    "THRESHOLDS",
    "pick_best",
    "score_grid",
    "update_sidecar",
]


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

    report = tune_report(results, best, refractory_frames, cfg)
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
