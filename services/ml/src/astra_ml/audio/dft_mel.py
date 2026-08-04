"""Log-mel frontend built from fixed-weight matmuls, so it lives inside the ONNX graph.

With window == hop == frame size, the STFT degenerates to one DFT per frame:
only MatMul/Mul/Add/Log ops — no STFT operator, nothing export-fragile.
"""

import math

import torch
import torchaudio
from torch import nn

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320  # 20 ms @ 16 kHz
N_FFT = FRAME_SAMPLES
N_BINS = N_FFT // 2 + 1  # 161
N_MELS = 40
LOG_EPS = 1e-6


class MelFrontend(nn.Module):
    """[B, n_frames * 320] PCM in [-1, 1] → [B, n_frames, 40] log-mel."""

    def __init__(self) -> None:
        super().__init__()
        window = torch.hann_window(N_FFT, periodic=True)
        n = torch.arange(N_FFT, dtype=torch.float64)
        k = torch.arange(N_BINS, dtype=torch.float64)
        angle = 2 * math.pi * torch.outer(k, n) / N_FFT
        # Window folded into the DFT basis: X_k = Σ_n x_n · w_n · e^{-i·angle}
        self.register_buffer("cos_basis", (torch.cos(angle) * window.double()).T.float())
        self.register_buffer("sin_basis", (torch.sin(angle) * window.double()).T.float())
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=N_BINS,
            f_min=0.0,
            f_max=SAMPLE_RATE / 2,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
        )
        self.register_buffer("mel_fb", mel_fb)

    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        frames = pcm.reshape(pcm.shape[0], -1, FRAME_SAMPLES)
        real = frames @ self.cos_basis  # [B, F, 161]
        imag = frames @ self.sin_basis
        power = real * real + imag * imag
        mel = power @ self.mel_fb  # [B, F, 40]
        return torch.log(mel + LOG_EPS)


# Wake-word v2 (BC-ResNet) frontend constants: the paper's 30 ms window / 10 ms shift.
KWS_WIN_SAMPLES = 480  # 30 ms @ 16 kHz
KWS_HOP_SAMPLES = 160  # 10 ms
KWS_N_MELS = 40  # load-bearing: BC-ResNet's freq path is 40 -> 20 -> 10 -> 5 -> 1
KWS_WINDOW_SAMPLES = 15_360  # 0.96 s = 12 x 1280-sample chunks
INT16_SCALE = 32_768.0


def kws_n_frames(n_samples: int = KWS_WINDOW_SAMPLES) -> int:
    return (n_samples - KWS_WIN_SAMPLES) // KWS_HOP_SAMPLES + 1


class LogMelSTFT(nn.Module):
    """[B, n_samples] PCM → [B, 1, n_mels, n_frames] log-mel, ready for BC-ResNet.

    Unlike MelFrontend the window overlaps the hop, so framing can no longer be a
    reshape. It is a Gather against a constant index matrix instead — still only
    Gather/MatMul/Mul/Add/Log in the exported graph, nothing STFT-shaped.

    input_scale is folded into the DFT basis, so the graph consumes raw int16 sample
    values as float32 (the sidecar's `input_scale: "int16"` contract that the Go
    runtime already feeds) while the maths matches PCM in [-1, 1].
    """

    def __init__(
        self,
        n_samples: int = KWS_WINDOW_SAMPLES,
        win_samples: int = KWS_WIN_SAMPLES,
        hop_samples: int = KWS_HOP_SAMPLES,
        n_mels: int = KWS_N_MELS,
        input_scale: float = INT16_SCALE,
    ) -> None:
        super().__init__()
        self.n_frames = (n_samples - win_samples) // hop_samples + 1
        self.n_mels = n_mels
        n_bins = win_samples // 2 + 1

        starts = hop_samples * torch.arange(self.n_frames)
        offsets = torch.arange(win_samples)
        self.register_buffer("frame_idx", (starts[:, None] + offsets[None, :]).long())

        window = torch.hann_window(win_samples, periodic=True).double() / input_scale
        n = torch.arange(win_samples, dtype=torch.float64)
        k = torch.arange(n_bins, dtype=torch.float64)
        angle = 2 * math.pi * torch.outer(k, n) / win_samples
        # Window and input scaling folded into the DFT basis: both are per-sample
        # multipliers, so they cost nothing at inference.
        self.register_buffer("cos_basis", (torch.cos(angle) * window).T.float())
        self.register_buffer("sin_basis", (torch.sin(angle) * window).T.float())

        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_bins,
            f_min=0.0,
            f_max=SAMPLE_RATE / 2,
            n_mels=n_mels,
            sample_rate=SAMPLE_RATE,
        )
        self.register_buffer("mel_fb", mel_fb)

    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        frames = pcm[:, self.frame_idx]  # [B, n_frames, win]
        real = frames @ self.cos_basis  # [B, n_frames, n_bins]
        imag = frames @ self.sin_basis
        power = real * real + imag * imag
        mel = power @ self.mel_fb  # [B, n_frames, n_mels]
        mel = torch.log(mel + LOG_EPS)
        return mel.transpose(1, 2).unsqueeze(1)  # [B, 1, n_mels, n_frames]
