"""Wake-word export: head ONNX + single-graph merge + sidecar (ww_v1).

Primary artifact is one merged graph — audio [1, window_samples] of int16-range
float32 in, prob [1, 1] out — stitched from the two frozen OWW frontends, glue
(mel transform x/10+2, window unfold, reshapes) and the trained head, all at
opset 13 (the frontends' opset; the head is exported to match, so no version
conversion is involved). A parity gate scores random windows through the
merged graph vs the Python reference (WwFrontend + torch head); on failure the
export falls back to the chain-of-3 layout behind the same sidecar
("graph": "chain") without ceremony.

    uv run python -m astra_ml.export.export_ww --checkpoint runs/ww/best.pt
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from astra_ml.audio.ww_frontend import DEFAULT_FRONTEND, FrontendConfig, WwFrontend
from astra_ml.models.ww import WwHead
from astra_ml.training.train_ww import load_head
from astra_ml.training.ww_config import load_ww_config

OPSET = 13  # pinned to the frozen frontends
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUT = REPO_ROOT / "assets" / "models" / "wakeword"
PARITY_WINDOWS = 200
PARITY_TOL = 1e-4
SR = 16000
FRAME_SAMPLES = 320


class _ExportHead(torch.nn.Module):
    """Fixes the head ONNX contract: emb [1, F, D] -> prob [1, 1]."""

    def __init__(self, head: WwHead):
        super().__init__()
        self.head = head

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head.logits(emb)).reshape(1, 1)


def export_head(head: WwHead, path: Path, cfg: FrontendConfig) -> None:
    torch.onnx.export(
        _ExportHead(head).eval(),
        (torch.zeros(1, cfg.head_frames, cfg.emb_dim),),
        str(path),
        input_names=["emb"],
        output_names=["prob"],
        opset_version=OPSET,
        dynamo=False,
    )


def _glue_pre_embedding(cfg: FrontendConfig, mel_out_name: str) -> onnx.ModelProto:
    """mel output [1, 1, T, bins] -> transform -> unfold -> [head_frames, 76, bins, 1]."""
    t, bins = cfg.window_frames_total, cfg.mel_bins
    indices = np.stack(
        [
            np.arange(i * cfg.emb_stride, i * cfg.emb_stride + cfg.emb_window)
            for i in range(cfg.head_frames)
        ]
    ).astype(np.int64)
    nodes = [
        helper.make_node("Reshape", [mel_out_name, "g1_shape_flat"], ["g1_flat"]),
        helper.make_node("Mul", ["g1_flat", "g1_scale"], ["g1_scaled"]),
        helper.make_node("Add", ["g1_scaled", "g1_offset"], ["g1_mel"]),
        helper.make_node("Gather", ["g1_mel", "g1_indices"], ["g1_windows"], axis=0),
        helper.make_node("Reshape", ["g1_windows", "g1_shape_out"], ["g1_out"]),
    ]
    initializers = [
        numpy_helper.from_array(np.array([t, bins], dtype=np.int64), "g1_shape_flat"),
        numpy_helper.from_array(np.array(cfg.transform_scale, dtype=np.float32), "g1_scale"),
        numpy_helper.from_array(np.array(cfg.transform_offset, dtype=np.float32), "g1_offset"),
        numpy_helper.from_array(indices, "g1_indices"),
        numpy_helper.from_array(
            np.array([cfg.head_frames, cfg.emb_window, bins, 1], dtype=np.int64), "g1_shape_out"
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "ww_glue_pre_embedding",
        [helper.make_tensor_value_info(mel_out_name, TensorProto.FLOAT, [1, 1, t, bins])],
        [
            helper.make_tensor_value_info(
                "g1_out", TensorProto.FLOAT, [cfg.head_frames, cfg.emb_window, bins, 1]
            )
        ],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def _glue_pre_head(cfg: FrontendConfig, emb_out_name: str) -> onnx.ModelProto:
    """embedding output [head_frames, 1, 1, dim] -> [1, head_frames, dim]."""
    nodes = [helper.make_node("Reshape", [emb_out_name, "g2_shape"], ["g2_out"])]
    initializers = [
        numpy_helper.from_array(
            np.array([1, cfg.head_frames, cfg.emb_dim], dtype=np.int64), "g2_shape"
        )
    ]
    graph = helper.make_graph(
        nodes,
        "ww_glue_pre_head",
        [
            helper.make_tensor_value_info(
                emb_out_name, TensorProto.FLOAT, [cfg.head_frames, 1, 1, cfg.emb_dim]
            )
        ],
        [
            helper.make_tensor_value_info(
                "g2_out", TensorProto.FLOAT, [1, cfg.head_frames, cfg.emb_dim]
            )
        ],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def _rename_graph_io(model: onnx.ModelProto, renames: dict[str, str]) -> None:
    for value_info in list(model.graph.input) + list(model.graph.output):
        if value_info.name in renames:
            value_info.name = renames[value_info.name]
    for node in model.graph.node:
        for i, n in enumerate(node.input):
            node.input[i] = renames.get(n, n)
        for i, n in enumerate(node.output):
            node.output[i] = renames.get(n, n)


def merge(
    melspec_path: Path, embedding_path: Path, head_path: Path, cfg: FrontendConfig
) -> onnx.ModelProto:
    from onnx import compose

    melspec = compose.add_prefix(onnx.load(melspec_path), "mel_")
    embedding = compose.add_prefix(onnx.load(embedding_path), "emb_")
    head = compose.add_prefix(onnx.load(head_path), "head_")

    mel_out = melspec.graph.output[0].name
    emb_in = embedding.graph.input[0].name
    emb_out = embedding.graph.output[0].name
    head_in = head.graph.input[0].name
    head_out = head.graph.output[0].name

    glue1 = _glue_pre_embedding(cfg, mel_out)
    glue2 = _glue_pre_head(cfg, emb_out)

    models = (melspec, embedding, head, glue1, glue2)
    ir_version = max(m.ir_version for m in models)
    for m in models:
        m.ir_version = ir_version  # merge_models demands equal IR versions
        kept = [op for op in m.opset_import if op.domain in ("", "ai.onnx.ml")]
        del m.opset_import[:]
        m.opset_import.extend(kept)

    merged = compose.merge_models(melspec, glue1, io_map=[(mel_out, mel_out)])
    merged = compose.merge_models(merged, embedding, io_map=[("g1_out", emb_in)])
    merged = compose.merge_models(merged, glue2, io_map=[(emb_out, emb_out)])
    merged = compose.merge_models(merged, head, io_map=[("g2_out", head_in)])

    _rename_graph_io(merged, {melspec.graph.input[0].name: "audio", head_out: "prob"})
    # pin the input shape: [1, window_samples]
    dims = merged.graph.input[0].type.tensor_type.shape.dim
    dims[0].dim_value = 1
    dims[1].dim_value = cfg.window_samples
    onnx.checker.check_model(merged)
    return merged


def parity_check(
    merged_path: Path,
    frontends_dir: Path,
    head: WwHead,
    cfg: FrontendConfig,
    n_windows: int = PARITY_WINDOWS,
) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(merged_path))
    frontend = WwFrontend.from_onnx(frontends_dir, cfg)
    export_head_module = _ExportHead(head).eval()
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(n_windows):
        audio = (rng.uniform(-0.5, 0.5, cfg.window_samples) * 32767).astype(np.float32)
        with torch.no_grad():
            want = export_head_module(
                torch.from_numpy(frontend.features(audio).astype(np.float32))[None]
            ).numpy()
        got = session.run(None, {"audio": audio[None]})[0]
        max_diff = max(max_diff, float(np.abs(got - want).max()))
    return max_diff


def sidecar_dict(
    graph: str, threshold: float, postproc: dict, gating: dict, cfg: FrontendConfig
) -> dict:
    return {
        "sample_rate": SR,
        "graph": graph,
        "chunk_samples": cfg.chunk_samples,
        "window_samples": cfg.window_samples,
        "input_scale": "int16",  # raw int16 sample values as float32, NOT [-1,1]
        "mel": {
            "bins": cfg.mel_bins,
            "hop_samples": cfg.mel_hop,
            "window_samples": cfg.mel_window,
            "lookback_samples": cfg.mel_lookback,
            "frames_per_chunk": cfg.mel_frames_per_chunk,
            "window_frames_total": cfg.window_frames_total,
            "transform": {"scale": cfg.transform_scale, "offset": cfg.transform_offset},
        },
        "embedding": {
            "window_frames": cfg.emb_window,
            "stride_frames": cfg.emb_stride,
            "dim": cfg.emb_dim,
        },
        "head": {"input_frames": cfg.head_frames},
        "io": {
            "merged": {"input": "audio", "input_shape": [1, cfg.window_samples], "output": "prob"},
            "chain": {
                "melspec": {"file": "ww_v1_melspec.onnx", "input": "input", "output": "output"},
                "embedding": {
                    "file": "ww_v1_embedding.onnx",
                    "input": "input_1",
                    "output": "conv2d_19",
                },
                "head": {"file": "ww_v1_head.onnx", "input": "emb", "output": "prob"},
            },
        },
        "recommended_threshold": threshold,
        "postproc": postproc,
        "gating": gating,
        "opset": OPSET,
    }


def export(
    head: WwHead,
    frontends_dir: Path,
    out_dir: Path,
    threshold: float,
    postproc: dict,
    gating: dict,
    cfg: FrontendConfig = DEFAULT_FRONTEND,
    name: str = "ww_v1",
) -> str:
    """Returns the graph mode written: "merged" or "chain"."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = out_dir / f"{name}_head.onnx"
    export_head(head, head_path, cfg)

    melspec_path = Path(frontends_dir) / "melspectrogram.onnx"
    embedding_path = Path(frontends_dir) / "embedding_model.onnx"

    graph_mode = "merged"
    try:
        merged = merge(melspec_path, embedding_path, head_path, cfg)
        merged_path = out_dir / f"{name}.onnx"
        onnx.save(merged, merged_path)
        diff = parity_check(merged_path, frontends_dir, head, cfg)
        if diff >= PARITY_TOL:
            raise RuntimeError(f"parity gate failed: max diff {diff:.2e} >= {PARITY_TOL}")
        print(f"merged parity: max diff {diff:.2e} over {PARITY_WINDOWS} windows")
        head_path.unlink()  # merged mode ships a single model file
    except Exception as e:  # noqa: BLE001 — any merge failure drops to the chain layout
        print(f"merge failed ({e}); falling back to chain-of-3")
        graph_mode = "chain"
        (out_dir / f"{name}.onnx").unlink(missing_ok=True)
        shutil.copy(melspec_path, out_dir / f"{name}_melspec.onnx")
        shutil.copy(embedding_path, out_dir / f"{name}_embedding.onnx")

    sidecar = sidecar_dict(graph_mode, threshold, postproc, gating, cfg)
    (out_dir / f"{name}.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"exported ({graph_mode}) to {out_dir}")
    return graph_mode


def load_tuned(report_path: Path) -> tuple[float, int]:
    best = json.loads(report_path.read_text())["best"]
    return best["threshold"], best["patience_frames"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/ww/best.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v1.yaml"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tune-report", type=Path, default=Path("runs/ww/tune_report.json"))
    args = parser.parse_args()
    cfg = load_ww_config(args.config)

    threshold, patience = cfg.postproc.threshold, cfg.postproc.patience_frames
    if args.tune_report.exists():
        threshold, patience = load_tuned(args.tune_report)
        print(f"using tuned threshold {threshold}, patience {patience}")

    postproc = {
        "patience_frames": patience,
        "refractory_frames": int(cfg.postproc.refractory_seconds * SR / FRAME_SAMPLES),
    }
    gating = {
        "preroll_frames": cfg.gating.preroll_frames,
        "partial_chunk": cfg.gating.partial_chunk,
    }
    head = load_head(args.checkpoint)
    export(head, cfg.data.frontends_dir, args.out, threshold, postproc, gating)


if __name__ == "__main__":
    main()
