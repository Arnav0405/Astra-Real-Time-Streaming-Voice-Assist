"""Export checks for wake-word v2: real ONNX, real onnxruntime, untrained weights."""

import json

import numpy as np
import onnxruntime as ort
import torch
from test_kws_dataset import _cfg, corpus  # noqa: F401  (fixture reuse)

from astra_ml.export.export_kws import (
    OPSET,
    PARITY_TOL,
    build,
    export,
    parity_check,
    sidecar_dict,
)
from astra_ml.models.bcresnet import BCResNets


def _model(cfg):
    torch.manual_seed(0)
    model = BCResNets(base_c=cfg.model.base_c, num_classes=1)
    # Untrained BatchNorm running stats are the identity-ish defaults, which is fine:
    # this checks the graph, not the weights.
    return model.eval()


def test_export_produces_the_go_runtime_contract(corpus, tmp_path):  # noqa: F811
    cfg = _cfg(corpus)
    path = export(cfg, _model(cfg), tmp_path, threshold=0.85, patience=2)
    assert path.exists()

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (inp,) = session.get_inputs()
    (out,) = session.get_outputs()
    assert inp.name == "audio" and inp.shape == [1, cfg.frontend.window_samples]
    assert out.name == "prob" and out.shape == [1, 1]

    rng = np.random.default_rng(1)
    window = (rng.uniform(-0.4, 0.4, cfg.frontend.window_samples) * 32768).astype(np.float32)
    prob = session.run(["prob"], {"audio": window[None, :]})[0]
    assert prob.shape == (1, 1)
    assert 0.0 <= float(prob[0, 0]) <= 1.0


def test_export_parity_is_well_inside_the_gate(corpus, tmp_path):  # noqa: F811
    cfg = _cfg(corpus)
    graph = build(cfg, _model(cfg))
    path = tmp_path / "ww_v2.onnx"
    export(cfg, _model(cfg), tmp_path, threshold=0.85, patience=2)
    assert parity_check(graph, path, cfg.frontend.window_samples, seed=3) < PARITY_TOL


def test_sidecar_matches_what_the_go_loader_requires(corpus, tmp_path):  # noqa: F811
    cfg = _cfg(corpus)
    export(cfg, _model(cfg), tmp_path, threshold=0.71, patience=3)
    sidecar = json.loads((tmp_path / "ww_v2.json").read_text())

    # config.go:48-56 rejects the model unless all of these hold.
    assert sidecar["graph"] == "merged"
    assert sidecar["chunk_samples"] == 1280
    assert sidecar["window_samples"] == cfg.frontend.window_samples
    assert sidecar["recommended_threshold"] == 0.71
    assert sidecar["gating"]["partial_chunk"] == "drop"
    assert sidecar["io"]["merged"] == {
        "input": "audio",
        "input_shape": [1, cfg.frontend.window_samples],
        "output": "prob",
    }
    # refractory_seconds is authored in seconds and shipped in 20 ms frames.
    assert sidecar["postproc"] == {"patience_frames": 3, "refractory_frames": 100}
    assert sidecar["input_scale"] == "int16"
    assert sidecar["opset"] == OPSET


def test_sidecar_records_the_frame_count(corpus):  # noqa: F811
    cfg = _cfg(corpus)
    sidecar = sidecar_dict(cfg, threshold=0.85, patience=2, n_frames=94)
    assert sidecar["mel"] == {
        "bins": 40,
        "hop_samples": 160,
        "window_samples": 480,
        "window_frames_total": 94,
    }
    assert sidecar["model"]["arch"] == "bcresnet"


def test_exported_graph_is_deterministic_across_calls(corpus, tmp_path):  # noqa: F811
    """Dropout and BatchNorm must be folded to eval behaviour, not sampled per run."""
    cfg = _cfg(corpus)
    path = export(cfg, _model(cfg), tmp_path, threshold=0.85, patience=2)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    window = np.full((1, cfg.frontend.window_samples), 1000.0, dtype=np.float32)
    first = session.run(["prob"], {"audio": window})[0]
    second = session.run(["prob"], {"audio": window})[0]
    np.testing.assert_array_equal(first, second)
