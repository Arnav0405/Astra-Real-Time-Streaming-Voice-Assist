"""Dataset-free tests for the merged export using tiny stand-in frontends.

The stand-ins are real ONNX models with the same IO contract as the frozen
OWW frontends (names, ranks, opset 13) but toy sizes, so the whole
merge + parity-gate + sidecar path runs without downloads.
"""

import json

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from astra_ml.audio.ww_frontend import FrontendConfig
from astra_ml.export import export_ww
from astra_ml.models.ww import WwHead

TINY = FrontendConfig(
    chunk_samples=8,
    mel_hop=2,
    mel_window=4,
    mel_bins=3,
    mel_lookback=2,
    mel_frames_per_chunk=3,
    emb_window=4,
    emb_stride=2,
    emb_dim=5,
    head_frames=3,
)
# window_frames_total = 4 + 2*2 = 8; window_samples = 7*2 + 4 = 18
T, BINS, DIM = 8, 3, 5


def standin_melspec(path):
    """input [1, 18] -> MatMul -> output [1, 1, 8, 3] (names match the real model)."""
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.normal(0, 0.1, (18, T * BINS)).astype(np.float32), "mel_w")
    nodes = [
        helper.make_node("MatMul", ["input", "mel_w"], ["mm"]),
        helper.make_node("Reshape", ["mm", "mel_shape"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "standin_melspec",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 18])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, T, BINS])],
        [w, numpy_helper.from_array(np.array([1, 1, T, BINS], dtype=np.int64), "mel_shape")],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def standin_embedding(path):
    """input_1 [N, 4, 3, 1] -> MatMul -> conv2d_19 [N, 1, 1, 5]."""
    rng = np.random.default_rng(1)
    w = numpy_helper.from_array(rng.normal(0, 0.1, (12, DIM)).astype(np.float32), "emb_w")
    nodes = [
        helper.make_node("Reshape", ["input_1", "emb_flat"], ["flat"]),
        helper.make_node("MatMul", ["flat", "emb_w"], ["mm"]),
        helper.make_node("Reshape", ["mm", "emb_shape"], ["conv2d_19"]),
    ]
    graph = helper.make_graph(
        nodes,
        "standin_embedding",
        [helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [None, 4, BINS, 1])],
        [helper.make_tensor_value_info("conv2d_19", TensorProto.FLOAT, [None, 1, 1, DIM])],
        [
            w,
            numpy_helper.from_array(np.array([-1, 12], dtype=np.int64), "emb_flat"),
            numpy_helper.from_array(np.array([-1, 1, 1, DIM], dtype=np.int64), "emb_shape"),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


@pytest.fixture
def frontends_dir(tmp_path):
    d = tmp_path / "frontends"
    d.mkdir()
    standin_melspec(d / "melspectrogram.onnx")
    standin_embedding(d / "embedding_model.onnx")
    return d


@pytest.fixture
def head():
    torch.manual_seed(0)
    return WwHead(layer_size=8, head_frames=TINY.head_frames, emb_dim=TINY.emb_dim)


POSTPROC = {"patience_frames": 2, "refractory_frames": 100}
GATING = {"preroll_frames": 50, "partial_chunk": "drop"}


def test_export_merged(frontends_dir, head, tmp_path, capsys):
    out = tmp_path / "out"
    mode = export_ww.export(head, frontends_dir, out, 0.5, POSTPROC, GATING, TINY, name="ww_t")
    assert mode == "merged"
    assert (out / "ww_t.onnx").exists()
    assert not (out / "ww_t_head.onnx").exists()  # single-file artifact

    sidecar = json.loads((out / "ww_t.json").read_text())
    assert sidecar["graph"] == "merged"
    assert sidecar["window_samples"] == TINY.window_samples
    assert sidecar["input_scale"] == "int16"
    assert sidecar["recommended_threshold"] == 0.5
    assert sidecar["postproc"] == POSTPROC
    assert sidecar["gating"] == GATING
    assert sidecar["mel"]["transform"] == {"scale": 0.1, "offset": 2.0}
    assert sidecar["io"]["merged"] == {
        "input": "audio",
        "input_shape": [1, TINY.window_samples],
        "output": "prob",
    }


def test_merged_model_runs_and_matches_reference(frontends_dir, head, tmp_path):
    import onnxruntime as ort

    from astra_ml.audio.ww_frontend import WwFrontend

    out = tmp_path / "out"
    export_ww.export(head, frontends_dir, out, 0.5, POSTPROC, GATING, TINY, name="ww_t")
    session = ort.InferenceSession(str(out / "ww_t.onnx"))
    frontend = WwFrontend.from_onnx(frontends_dir, TINY)

    rng = np.random.default_rng(42)
    audio = (rng.uniform(-0.5, 0.5, TINY.window_samples) * 32767).astype(np.float32)
    got = session.run(None, {"audio": audio[None]})[0]
    with torch.no_grad():
        feats = torch.from_numpy(frontend.features(audio).astype(np.float32))[None]
        want = torch.sigmoid(head.logits(feats)).reshape(1, 1).numpy()
    np.testing.assert_allclose(got, want, atol=1e-4)
    assert got.shape == (1, 1)


def test_export_falls_back_to_chain(frontends_dir, head, tmp_path, monkeypatch, capsys):
    def broken_merge(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(export_ww, "merge", broken_merge)
    out = tmp_path / "out"
    mode = export_ww.export(head, frontends_dir, out, 0.5, POSTPROC, GATING, TINY, name="ww_t")
    assert mode == "chain"
    assert not (out / "ww_t.onnx").exists()
    for suffix in ("melspec", "embedding", "head"):
        assert (out / f"ww_t_{suffix}.onnx").exists()
    sidecar = json.loads((out / "ww_t.json").read_text())
    assert sidecar["graph"] == "chain"
    assert "falling back to chain-of-3" in capsys.readouterr().out


def test_export_chain_when_parity_fails(frontends_dir, head, tmp_path, monkeypatch):
    monkeypatch.setattr(export_ww, "parity_check", lambda *a, **k: 1.0)
    out = tmp_path / "out"
    mode = export_ww.export(head, frontends_dir, out, 0.5, POSTPROC, GATING, TINY, name="ww_t")
    assert mode == "chain"


def test_load_tuned(tmp_path):
    report = tmp_path / "tune_report.json"
    report.write_text(json.dumps({"best": {"threshold": 0.65, "patience_frames": 3}}))
    assert export_ww.load_tuned(report) == (0.65, 3)
