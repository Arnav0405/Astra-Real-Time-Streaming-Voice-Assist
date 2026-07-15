"""Export the VAD model to ONNX + sidecar metadata for the Go runtime.

Usage:
    uv run python -m astra_ml.export.export [--checkpoint runs/vad/best.pt] \
        [--out ../../assets/models/vad] [--threshold 0.5]
"""

import argparse
import json
from pathlib import Path

import torch

from astra_ml.models.vad import FRAME_SAMPLES, HIDDEN_SIZE, StreamingVad, VadModel, zero_state

OPSET = 17
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUT = REPO_ROOT / "assets" / "models" / "vad"


def export(model: VadModel, out_dir: Path, threshold: float, name: str = "vad_v1") -> Path:
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
        "opset": OPSET,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = VadModel()
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    path = export(model, args.out, args.threshold)
    print(f"exported {path}")


if __name__ == "__main__":
    main()
