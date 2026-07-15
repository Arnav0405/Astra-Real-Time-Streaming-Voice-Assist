import json

import numpy as np
import onnxruntime
import torch

from astra_ml.export.export import export
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
