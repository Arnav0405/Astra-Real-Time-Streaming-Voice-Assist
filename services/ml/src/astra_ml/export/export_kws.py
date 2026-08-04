"""Wake-word v2 export: one ONNX graph (log-mel + BC-ResNet + sigmoid) plus sidecar.

    uv run python -m astra_ml.export.export_kws --checkpoint runs/ww_v2/best.pt

v1 needed 300 lines of hand-stitched ONNX because its frontend was two frozen third-party
graphs that had to be glued to the trained head. v2 trains everything below the sigmoid,
so the whole thing is a single torch.onnx.export. The parity gate stays: 200 random
windows through torch vs onnxruntime, because "the exported graph computes what training
computed" is worth a hard check regardless of how the graph was built.

Contract, unchanged from v1 so the Go runtime needs no source change:
    audio [1, window_samples] float32, raw int16 sample values  ->  prob [1, 1]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from astra_ml.audio.dft_mel import LogMelSTFT
from astra_ml.models.bcresnet import BCResNets
from astra_ml.training.kws_config import KwsConfig, load_kws_config
from astra_ml.training.train_kws import load_model

OPSET = 17  # matches the VAD model already running in this Go runtime
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUT = REPO_ROOT / "assets" / "models" / "wakeword"
PARITY_WINDOWS = 200
PARITY_TOL = 1e-4
SR = 16000
FRAME_SAMPLES = 320  # 20 ms transport frame; refractory is counted in these


class KwsWakeWord(nn.Module):
    """The shipped graph: waveform in, probability out, nothing in between."""

    def __init__(self, frontend: LogMelSTFT, model: BCResNets) -> None:
        super().__init__()
        self.frontend = frontend
        self.model = model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(self.frontend(audio)))


def build(cfg: KwsConfig, model: BCResNets) -> KwsWakeWord:
    frontend = LogMelSTFT(
        n_samples=cfg.frontend.window_samples,
        win_samples=cfg.frontend.win_samples,
        hop_samples=cfg.frontend.hop_samples,
        n_mels=cfg.frontend.n_mels,
    )
    # eval() matters more than usual here: BatchNorm and SubSpectralNorm both fold to
    # their running statistics, and exporting in train mode would bake batch statistics
    # of a single dummy window into the shipped graph.
    return KwsWakeWord(frontend, model).eval()


def export_onnx(graph: KwsWakeWord, path: Path, window_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        graph,
        torch.zeros(1, window_samples),
        str(path),
        input_names=["audio"],
        output_names=["prob"],
        opset_version=OPSET,
        # ponytail: the TorchScript exporter, deprecated but present. The dynamo exporter
        # needs onnxscript, which is not a dependency, and the graph here is static-shape
        # MatMul/Conv with no control flow — nothing the new path would do better.
        dynamo=False,
    )


def parity_check(graph: KwsWakeWord, path: Path, window_samples: int, seed: int = 0) -> float:
    """Max |torch - onnxruntime| over random int16-range windows."""
    rng = np.random.default_rng(seed)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for _ in range(PARITY_WINDOWS):
        window = (rng.uniform(-0.5, 0.5, window_samples) * 32768).astype(np.float32)
        with torch.no_grad():
            expected = graph(torch.from_numpy(window[None, :])).numpy()
        actual = session.run(["prob"], {"audio": window[None, :]})[0]
        worst = max(worst, float(np.abs(expected - actual).max()))
    return worst


def sidecar_dict(cfg: KwsConfig, threshold: float, patience: int, n_frames: int) -> dict:
    f = cfg.frontend
    return {
        "sample_rate": SR,
        # The Go loader rejects anything but "merged" (config.go:48-49). v2 has no chain
        # layout at all — there is only ever one file.
        "graph": "merged",
        "chunk_samples": f.chunk_samples,
        "window_samples": f.window_samples,
        "input_scale": "int16",  # raw int16 sample values as float32, NOT [-1,1]
        "mel": {
            "bins": f.n_mels,
            "hop_samples": f.hop_samples,
            "window_samples": f.win_samples,
            "window_frames_total": n_frames,
        },
        "model": {
            "arch": "bcresnet",
            "base_c": cfg.model.base_c,
            "sub_bands": cfg.model.sub_bands,
        },
        "io": {
            "merged": {"input": "audio", "input_shape": [1, f.window_samples], "output": "prob"}
        },
        "recommended_threshold": threshold,
        "postproc": {
            "patience_frames": patience,
            "refractory_frames": int(cfg.postproc.refractory_seconds * SR / FRAME_SAMPLES),
        },
        "gating": {
            "preroll_frames": cfg.gating.preroll_frames,
            "partial_chunk": cfg.gating.partial_chunk,
        },
        "opset": OPSET,
    }


def export(
    cfg: KwsConfig,
    model: BCResNets,
    out_dir: Path,
    threshold: float,
    patience: int,
    name: str = "ww_v2",
) -> Path:
    graph = build(cfg, model)
    model_path = out_dir / f"{name}.onnx"
    export_onnx(graph, model_path, cfg.frontend.window_samples)

    diff = parity_check(graph, model_path, cfg.frontend.window_samples)
    if diff >= PARITY_TOL:
        raise RuntimeError(f"parity gate failed: max diff {diff:.2e} >= {PARITY_TOL}")
    print(f"torch vs onnxruntime: max diff {diff:.2e} over {PARITY_WINDOWS} windows")

    sidecar = sidecar_dict(cfg, threshold, patience, graph.frontend.n_frames)
    sidecar_path = out_dir / f"{name}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"exported to {model_path} ({model_path.stat().st_size / 1e6:.2f} MB)")
    return model_path


def load_tuned(report_path: Path) -> tuple[float, int]:
    best = json.loads(report_path.read_text())["best"]
    return best["threshold"], best["patience_frames"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/ww_v2/best.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v2.yaml"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tune-report", type=Path, default=Path("runs/ww_v2/tune_report.json"))
    args = parser.parse_args()

    cfg = load_kws_config(args.config)
    threshold, patience = cfg.postproc.threshold, cfg.postproc.patience_frames
    if args.tune_report.exists():
        threshold, patience = load_tuned(args.tune_report)
        print(f"using tuned threshold {threshold}, patience {patience}")

    export(cfg, load_model(args.checkpoint), args.out, threshold, patience)


if __name__ == "__main__":
    main()
