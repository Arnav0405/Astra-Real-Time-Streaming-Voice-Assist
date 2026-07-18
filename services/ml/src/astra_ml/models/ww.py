"""Wake-word classifier head over frozen OpenWakeWord embeddings.

The frontends (melspectrogram + Google speech-embedding) are frozen ONNX
models; only this head trains. Input is the feature window produced by
astra_ml.audio.ww_frontend: [B, 16, 96] float32.
"""

import torch
from torch import nn

HEAD_FRAMES = 16
EMB_DIM = 96


class WwHead(nn.Module):
    def __init__(
        self, layer_size: int = 128, head_frames: int = HEAD_FRAMES, emb_dim: int = EMB_DIM
    ) -> None:
        super().__init__()
        self.layer_size = layer_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_frames * emb_dim, layer_size),
            nn.ReLU(),
            nn.LayerNorm(layer_size),
            nn.Linear(layer_size, 1),
        )

    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        """[B, 16, 96] features → [B] pre-sigmoid logits (training)."""
        return self.net(feats).squeeze(-1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """[B, 16, 96] features → [B] wake probability."""
        return torch.sigmoid(self.logits(feats))
