"""Export the VAD model to ONNX + sidecar metadata for the Go runtime.

If a tune report (from astra_ml.evaluation.tune_postproc) exists, its best
onset threshold and postproc knobs are written into the sidecar; otherwise
--threshold and DEFAULT_POSTPROC apply. The postproc state machine itself
stays out of the ONNX graph by design (decision #10) — the Go runtime owns it.

Usage:
    uv run python -m astra_ml.export.export [--checkpoint runs/vad/best.pt] \
        [--out ../../assets/models/vad] [--threshold 0.5] \
        [--tune-report runs/vad/tune_report.json]
"""

import argparse
import json
from pathlib import Path

import torch

from astra_ml.models.vad import FRAME_SAMPLES, HIDDEN_SIZE, StreamingVad, VadModel, zero_state

OPSET = 17
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUT = REPO_ROOT / "assets" / "models" / "vad"

# Consumed by astra_ml.postproc (Python reference) and the Go runtime state
# machine — retune via segment_eval, don't edit ad hoc. Frames are 20 ms.
DEFAULT_POSTPROC = {
    "offset_threshold": 0.45,
    "min_speech_frames": 3,  # 60 ms onset debounce
    "min_silence_frames": 25,  # 500 ms hangover
}


def load_tuned(report_path: Path) -> tuple[float, dict]:
    """Best (onset threshold, postproc knobs) from a tune_postproc report."""
    best = json.loads(report_path.read_text())["best"]
    return best["onset_threshold"], {k: best[k] for k in DEFAULT_POSTPROC}


def export(
    model: VadModel,
    out_dir: Path,
    threshold: float,
    postproc: dict | None = None,
    name: str = "vad_v1",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{name}.onnx"
    wrapper = StreamingVad(model).eval()
    torch.onnx.export(
        wrapper,
        (torch.zeros(1, FRAME_SAMPLES), zero_state()),
        str(onnx_path),
        input_names=["pcm", "state_in"],
        output_names=["prob", "state_out"],
        opset_version=OPSET,
        dynamo=False,
    )
    sidecar = {
        "sample_rate": 16_000,
        "frame_samples": FRAME_SAMPLES,
        "state_shape": [1, 1, HIDDEN_SIZE],
        "recommended_threshold": threshold,
        "postproc": postproc if postproc is not None else DEFAULT_POSTPROC,
        "opset": OPSET,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-report", type=Path, default=Path("runs/vad/tune_report.json"))
    args = parser.parse_args()

    if args.tune_report.exists():
        threshold, postproc = load_tuned(args.tune_report)
        print(f"tuned postproc ← {args.tune_report}")
    else:
        threshold, postproc = args.threshold, None
        print(f"no tune report at {args.tune_report}; using --threshold and defaults")

    model = VadModel()
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    path = export(model, args.out, threshold, postproc)
    print(f"exported {path}")


if __name__ == "__main__":
    main()
