import torch

from astra_ml.models.vad import FRAME_SAMPLES, VadModel, zero_state


def test_forward_shapes_and_range():
    model = VadModel().eval()
    probs = model(torch.rand(3, 20 * FRAME_SAMPLES) * 2 - 1)
    assert probs.shape == (3, 20)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_param_budget():
    n_params = sum(p.numel() for p in VadModel().parameters())
    assert n_params < 100_000, n_params


def test_streaming_matches_batch():
    torch.manual_seed(1)
    model = VadModel().eval()
    n_frames = 25
    pcm = torch.rand(1, n_frames * FRAME_SAMPLES) * 2 - 1

    with torch.no_grad():
        batch_probs = model(pcm)
        state = zero_state()
        stream_probs = []
        for i in range(n_frames):
            frame = pcm[:, i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            prob, state = model.stream_step(frame, state)
            stream_probs.append(prob)
        stream_probs = torch.cat(stream_probs, dim=1)

    torch.testing.assert_close(stream_probs, batch_probs, atol=1e-5, rtol=1e-5)
