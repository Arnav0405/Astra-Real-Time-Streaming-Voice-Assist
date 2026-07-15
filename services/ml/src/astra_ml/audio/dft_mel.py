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
