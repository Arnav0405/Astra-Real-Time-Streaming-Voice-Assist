"""VAD model: mel frontend → frequency CNN → GRU → sigmoid head.

Temporal context lives entirely in the GRU hidden state, so streaming
inference needs exactly one state tensor threaded between 20 ms frames.
"""

import torch
from torch import nn

from astra_ml.audio.dft_mel import FRAME_SAMPLES, N_MELS, MelFrontend

HIDDEN_SIZE = 64
FEATURE_SIZE = 64


class FreqCNN(nn.Module):
    """Per-frame feature extractor over the 40 mel bins: [B, F, 40] → [B, F, 64]."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.proj = nn.Linear(32 * (N_MELS // 4), FEATURE_SIZE)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        batch, n_frames, _ = mel.shape
        x = mel.reshape(batch * n_frames, 1, N_MELS)
        x = self.conv(x).flatten(1)
        return self.proj(x).reshape(batch, n_frames, FEATURE_SIZE)


class VadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frontend = MelFrontend()
        self.cnn = FreqCNN()
        self.gru = nn.GRU(FEATURE_SIZE, HIDDEN_SIZE, batch_first=True)
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def logits(self, pcm: torch.Tensor) -> torch.Tensor:
        """[B, n_frames * 320] PCM → [B, n_frames] pre-sigmoid logits (training)."""
        feats = self.cnn(self.frontend(pcm))
        out, _ = self.gru(feats)
        return self.head(out).squeeze(-1)

    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        """[B, n_frames * 320] PCM → [B, n_frames] speech probability."""
        return torch.sigmoid(self.logits(pcm))

    def stream_step(
        self, pcm: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One 20 ms frame: [B, 320] PCM + [1, B, 64] state → ([B, 1] prob, new state)."""
        feats = self.cnn(self.frontend(pcm))
        out, new_state = self.gru(feats, state)
        return torch.sigmoid(self.head(out)).squeeze(-1), new_state


class StreamingVad(nn.Module):
    """Export wrapper fixing the ONNX contract:

    (pcm [1, 320], state_in [1, 1, 64]) → (prob [1, 1], state_out [1, 1, 64])
    """

    def __init__(self, model: VadModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, pcm: torch.Tensor, state_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.stream_step(pcm, state_in)


def zero_state(batch: int = 1) -> torch.Tensor:
    return torch.zeros(1, batch, HIDDEN_SIZE)


__all__ = ["FRAME_SAMPLES", "HIDDEN_SIZE", "StreamingVad", "VadModel", "zero_state"]
