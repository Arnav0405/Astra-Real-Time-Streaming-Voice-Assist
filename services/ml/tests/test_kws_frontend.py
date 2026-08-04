"""Dataset-free checks for the wake-word v2 frontend and BC-ResNet shapes."""

import pytest
import torch
import torchaudio

from astra_ml.audio.dft_mel import (
    FRAME_SAMPLES,
    KWS_HOP_SAMPLES,
    KWS_N_MELS,
    KWS_WIN_SAMPLES,
    KWS_WINDOW_SAMPLES,
    LOG_EPS,
    SAMPLE_RATE,
    LogMelSTFT,
    MelFrontend,
    kws_n_frames,
)
from astra_ml.models.bcresnet import BCResNets
from astra_ml.models.subspectralnorm import SubSpectralNorm


def _noise(n_samples: int = KWS_WINDOW_SAMPLES, batch: int = 2) -> torch.Tensor:
    """Int16-range PCM, the scale the Go runtime feeds (engine.go:51-53)."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(batch, n_samples, generator=g) * 4000.0


def test_ssn_preserves_shape():
    ssn = SubSpectralNorm(8, 5)
    x = torch.randn(3, 8, 20, 94)
    assert ssn(x).shape == x.shape


@pytest.mark.parametrize("freq", [20, 10, 5])
def test_ssn_accepts_every_bcresnet_freq_stage(freq):
    """40 mel bins give freq 20/10/5 at the three SSN sites; all divide by 5."""
    ssn = SubSpectralNorm(4, 5)
    assert ssn(torch.randn(2, 4, freq, 30)).shape == (2, 4, freq, 30)


def test_ssn_rejects_indivisible_freq():
    ssn = SubSpectralNorm(4, 5)
    with pytest.raises(ValueError, match="not divisible"):
        ssn(torch.randn(2, 4, 16, 30))  # what 32 mel bins would produce


def test_logmel_matches_torchaudio():
    pcm = _noise()
    ours = LogMelSTFT()(pcm)

    ref_power = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=KWS_WIN_SAMPLES,
        win_length=KWS_WIN_SAMPLES,
        hop_length=KWS_HOP_SAMPLES,
        n_mels=KWS_N_MELS,
        f_min=0.0,
        f_max=SAMPLE_RATE / 2,
        center=False,
        power=2.0,
    )(pcm / 32768.0)
    ref = torch.log(ref_power + LOG_EPS).unsqueeze(1)

    assert ours.shape == ref.shape
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)


def test_logmel_shape_is_bcresnet_ready():
    out = LogMelSTFT()(_noise(batch=1))
    assert out.shape == (1, 1, KWS_N_MELS, kws_n_frames())
    assert kws_n_frames() == 94


def test_logmel_input_scale_is_int16():
    """1/32768 is folded into the basis, so int16-range in == [-1, 1] elsewhere."""
    pcm = _noise(batch=1)
    scaled = LogMelSTFT()(pcm)
    unscaled = LogMelSTFT(input_scale=1.0)(pcm / 32768.0)
    torch.testing.assert_close(scaled, unscaled, rtol=1e-5, atol=1e-5)


def test_vad_melfrontend_still_matches_torchaudio():
    """MelFrontend is the VAD's shipped, golden-pinned frontend — adding LogMelSTFT
    to this module must not perturb it."""
    pcm = _noise(n_samples=FRAME_SAMPLES * 20, batch=1) / 32768.0
    ours = MelFrontend()(pcm)
    ref = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=FRAME_SAMPLES,
        win_length=FRAME_SAMPLES,
        hop_length=FRAME_SAMPLES,
        n_mels=40,
        f_min=0.0,
        f_max=SAMPLE_RATE / 2,
        center=False,
        power=2.0,
    )(pcm)
    torch.testing.assert_close(ours, torch.log(ref + LOG_EPS).transpose(1, 2), rtol=1e-4, atol=1e-4)


def test_bcresnet_forward_shape():
    model = BCResNets(base_c=24, num_classes=1).eval()
    mel = LogMelSTFT()(_noise(batch=2))
    with torch.no_grad():
        out = model(mel)
    assert out.shape == (2, 1)


def test_bcresnet_frequency_path_collapses_to_one():
    """40 -> 20 (head) -> 10 (stage 1) -> 5 (stage 2) -> 1 (classifier)."""
    model = BCResNets(base_c=24, num_classes=1).eval()
    x = torch.randn(1, 1, KWS_N_MELS, 94)
    with torch.no_grad():
        x = model.cnn_head(x)
        assert x.shape[2] == 20
        seen = []
        for i, num_modules in enumerate(model.n):
            for j in range(num_modules):
                x = model.BCBlocks[i][j](x)
            seen.append(x.shape[2])
        assert seen == [20, 10, 5, 5]
        assert x.shape[3] == 94  # time is never strided


def test_bcresnet_time_length_is_free():
    """Global average pool over time means any window length works."""
    model = BCResNets(base_c=24, num_classes=1).eval()
    with torch.no_grad():
        for frames in (94, 126, 196):
            assert model(torch.randn(1, 1, KWS_N_MELS, frames)).shape == (1, 1)


def test_bcresnet_3_param_count():
    """BC-ResNet-3 is base_c = 8 * tau = 24; the paper reports 54.2k parameters."""
    params = sum(p.numel() for p in BCResNets(base_c=24, num_classes=1).parameters())
    assert 45_000 < params < 65_000
