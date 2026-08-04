"""SubSpectral Normalization (Chang et al., 2021), the norm BC-ResNet is built on.

Splits the frequency axis into S sub-bands and normalizes each independently, by
folding the sub-band index into the channel axis and handing the result to a plain
BatchNorm2d. Exports to ONNX as Reshape/BatchNormalization/Reshape.

Requires F % S == 0. With 40 mel bins the frequency path is 40 -> 20 -> 10 -> 5,
all divisible by S = 5.
"""

import torch
from torch import nn


class SubSpectralNorm(nn.Module):
    """[N, C, F, T] -> [N, C, F, T], normalized per (channel, sub-band)."""

    def __init__(self, channels: int, sub_bands: int = 5) -> None:
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, f, t = x.shape
        # Skipped while tracing: shapes are static in the exported graph, and reading one
        # as a Python int there is what raises TracerWarning.
        if not torch.jit.is_tracing() and f % self.sub_bands != 0:
            raise ValueError(f"frequency dim {f} not divisible by {self.sub_bands} sub-bands")
        x = x.view(n, c * self.sub_bands, f // self.sub_bands, t)
        x = self.bn(x)
        return x.view(n, c, f, t)


__all__ = ["SubSpectralNorm"]
