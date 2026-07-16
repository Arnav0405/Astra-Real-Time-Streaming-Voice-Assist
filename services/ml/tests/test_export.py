import json

import numpy as np
import onnxruntime
import torch

from astra_ml.export.export import DEFAULT_POSTPROC, export, load_tuned
from astra_ml.models.vad import FRAME_SAMPLES, VadModel, zero_state


def test_onnx_matches_pytorch_with_state_threading(tmp_path):
    torch.manual_seed(2)
    model = VadModel().eval()
    onnx_path = export(model, tmp_path, threshold=0.5)

    sidecar = json.loads((tmp_path / "vad_v1.json").read_text())
    assert sidecar["frame_samples"] == FRAME_SAMPLES

    session = onnxruntime.InferenceSession(str(onnx_path))
    assert [i.name for i in session.get_inputs()] == ["pcm", "state_in"]
    assert [o.name for o in session.get_outputs()] == ["prob", "state_out"]

    pt_state = zero_state()
    ort_state = np.zeros((1, 1, 64), dtype=np.float32)
    for i in range(20):
        frame = torch.rand(1, FRAME_SAMPLES) * 2 - 1
        with torch.no_grad():
            pt_prob, pt_state = model.stream_step(frame, pt_state)
        ort_prob, ort_state = session.run(None, {"pcm": frame.numpy(), "state_in": ort_state})
        assert ort_prob.shape == (1, 1)
        np.testing.assert_allclose(
            ort_prob, pt_prob.numpy(), atol=1e-4, rtol=1e-4, err_msg=f"frame {i}"
        )
    np.testing.assert_allclose(ort_state, pt_state.numpy(), atol=1e-4, rtol=1e-4)


def test_sidecar_carries_postproc(tmp_path):
    model = VadModel().eval()
    tuned = {"offset_threshold": 0.3, "min_speech_frames": 5, "min_silence_frames": 15}
    export(model, tmp_path, threshold=0.65, postproc=tuned)
    sidecar = json.loads((tmp_path / "vad_v1.json").read_text())
    assert sidecar["recommended_threshold"] == 0.65
    assert sidecar["postproc"] == tuned

    export(model, tmp_path, threshold=0.5)  # no tuning → defaults
    sidecar = json.loads((tmp_path / "vad_v1.json").read_text())
    assert sidecar["postproc"] == DEFAULT_POSTPROC


def test_load_tuned_reads_best_from_tune_report(tmp_path):
    report = {
        "best": {
            "onset_threshold": 0.6,
            "offset_threshold": 0.35,
            "min_speech_frames": 8,
            "min_silence_frames": 25,
            "segment_recall": 0.96,
        }
    }
    path = tmp_path / "tune_report.json"
    path.write_text(json.dumps(report))
    threshold, postproc = load_tuned(path)
    assert threshold == 0.6
    assert postproc == {
        "offset_threshold": 0.35,
        "min_speech_frames": 8,
        "min_silence_frames": 25,
    }
