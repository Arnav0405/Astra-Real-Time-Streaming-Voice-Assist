import torch
import torchaudio

from astra_ml.audio.dft_mel import (
    FRAME_SAMPLES,
    LOG_EPS,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    MelFrontend,
)


def test_matches_torchaudio_melspectrogram():
    torch.manual_seed(0)
    pcm = torch.rand(2, 50 * FRAME_SAMPLES) * 2 - 1

    reference = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=N_FFT,
        hop_length=N_FFT,
        n_mels=N_MELS,
        center=False,
        power=2.0,
    )(pcm)  # [B, 40, F]
    expected = torch.log(reference.transpose(1, 2) + LOG_EPS)

    got = MelFrontend()(pcm)
    assert got.shape == (2, 50, N_MELS)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def test_single_frame_shape():
    got = MelFrontend()(torch.zeros(1, FRAME_SAMPLES))
    assert got.shape == (1, 1, N_MELS)
