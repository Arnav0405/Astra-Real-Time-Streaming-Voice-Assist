"""Reference wake-word feature frontend — the arithmetic the Go runtime mirrors.

Wraps the two frozen OpenWakeWord ONNX models (melspectrogram, Google
speech-embedding) behind injectable callables so tests run without the
downloaded artifacts. All constants were pinned empirically against the real
models (astra_ml.data.oww_assets --inspect) and against
openwakeword.utils.AudioFeatures (parity test in tests/test_ww_frontend.py).

Audio convention: the melspectrogram model consumes float32 tensors holding raw
int16 PCM sample values (range ±32768), NOT [-1, 1] normalized audio. This
differs from the VAD model and is easy to get wrong on the Go side.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrontendConfig:
    chunk_samples: int = 1280  # one 80 ms score step = four 20 ms transport frames
    mel_hop: int = 160
    mel_window: int = 640  # effective window: frames(n) = (n - 640) // 160 + 1
    mel_bins: int = 32
    mel_lookback: int = 480  # raw samples of left context per streamed chunk (chain mode)
    mel_frames_per_chunk: int = 8
    transform_scale: float = 0.1  # spec/10 + 2, applied outside the ONNX graph
    transform_offset: float = 2.0
    emb_window: int = 76
    emb_stride: int = 8
    emb_dim: int = 96
    head_frames: int = 16

    @property
    def window_frames_total(self) -> int:
        return self.emb_window + (self.head_frames - 1) * self.emb_stride  # 196

    @property
    def window_samples(self) -> int:
        return (self.window_frames_total - 1) * self.mel_hop + self.mel_window  # 31840


DEFAULT_FRONTEND = FrontendConfig()

# melspec_fn: float32 [1, n] (int16-range values) -> raw mel squeezable to [frames, 32]
# embed_fn: float32 [batch, 76, 32, 1] -> embeddings squeezable to [batch, 96]
MelspecFn = Callable[[np.ndarray], np.ndarray]
EmbedFn = Callable[[np.ndarray], np.ndarray]


class WwFrontend:
    def __init__(
        self,
        melspec_fn: MelspecFn,
        embed_fn: EmbedFn,
        cfg: FrontendConfig = DEFAULT_FRONTEND,
    ):
        self.melspec_fn = melspec_fn
        self.embed_fn = embed_fn
        self.cfg = cfg

    def melspec(self, audio: np.ndarray) -> np.ndarray:
        """Transformed melspectrogram [frames, mel_bins] of a 1-D int16-range signal."""
        spec = np.asarray(self.melspec_fn(audio[None, :].astype(np.float32)))
        spec = spec.reshape(-1, self.cfg.mel_bins)
        return spec * self.cfg.transform_scale + self.cfg.transform_offset

    def features(self, audio: np.ndarray) -> np.ndarray:
        """[head_frames, emb_dim] features of exactly window_samples of audio."""
        cfg = self.cfg
        if audio.shape != (cfg.window_samples,):
            raise ValueError(f"expected shape ({cfg.window_samples},), got {audio.shape}")
        spec = self.melspec(audio)
        if spec.shape[0] != cfg.window_frames_total:
            raise ValueError(f"expected {cfg.window_frames_total} mel frames, got {spec.shape[0]}")
        windows = np.stack(
            [
                spec[i * cfg.emb_stride : i * cfg.emb_stride + cfg.emb_window]
                for i in range(cfg.head_frames)
            ]
        )
        emb = np.asarray(self.embed_fn(windows[:, :, :, None].astype(np.float32)))
        return emb.reshape(cfg.head_frames, cfg.emb_dim)

    @classmethod
    def from_onnx(cls, frontends_dir: Path, cfg: FrontendConfig = DEFAULT_FRONTEND) -> "WwFrontend":
        import onnxruntime as ort

        melspec = ort.InferenceSession(str(Path(frontends_dir) / "melspectrogram.onnx"))
        embedding = ort.InferenceSession(str(Path(frontends_dir) / "embedding_model.onnx"))
        mel_in = melspec.get_inputs()[0].name
        emb_in = embedding.get_inputs()[0].name
        return cls(
            melspec_fn=lambda x: melspec.run(None, {mel_in: x})[0],
            embed_fn=lambda x: embedding.run(None, {emb_in: x})[0],
            cfg=cfg,
        )
