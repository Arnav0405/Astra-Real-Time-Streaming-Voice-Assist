"""Wake-word head training: feature precompute + plain-PyTorch loop.

Usage (GPU machine, after data.ww_generate and data.ww_recordings):
    uv run python -m astra_ml.training.train_ww precompute --config configs/ww_v1.yaml
    uv run python -m astra_ml.training.train_ww train --config configs/ww_v1.yaml
Monitor: uv run tensorboard --logdir runs/ww

precompute runs every clip through the frozen frontends once and caches
float16 feature windows ([N, 16, 96]) in training.features_cache:

- positives.npy       TTS positives (augmented) + user train clips
                      (augment_rounds_user pitch/speed/noise variants each)
- positives_val.npy   TTS val positives, unaugmented
- negatives_adv.npy   adversarial TTS phrases
- negatives_local.npy windows slid over local negative audio dirs (optional)

train mixes those with the ~16 GB memory-mapped ACAV negative features
(data.acav_features; its tail is held out for validation) under a weighted BCE
whose negative weight ramps to training.max_negative_weight — the OWW recipe
for crushing false accepts.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from astra_ml.audio.ww_frontend import DEFAULT_FRONTEND, WwFrontend
from astra_ml.data.ww_augment import augment_clip, augment_user_clip, load_mono, scan_wavs
from astra_ml.models.ww import WwHead
from astra_ml.training.ww_config import WwConfig, load_ww_config

INT16_SCALE = 32767.0  # frontend consumes int16-range floats
JITTER_OFFSETS = (0, 800, 1600)  # word end at window end, -50 ms, -100 ms
ACAV_VAL_ROWS = 100_000
LOCAL_NEG_HOP_S = 1.0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def place_clip(clip: np.ndarray, end_offset: int, window_samples: int) -> np.ndarray:
    """Zero-pad/crop so the clip ends `end_offset` samples before the window end."""
    window = np.zeros(window_samples, dtype=np.float32)
    end = window_samples - end_offset
    start = max(0, end - len(clip))
    window[start:end] = clip[len(clip) - (end - start) :]
    return window


def clip_features(frontend: WwFrontend, clip: np.ndarray, jitters=JITTER_OFFSETS) -> np.ndarray:
    """[len(jitters), 16, 96] feature windows for one clip (float [-1,1] in)."""
    ws = frontend.cfg.window_samples
    out = [frontend.features(place_clip(clip, off, ws) * INT16_SCALE) for off in jitters]
    return np.stack(out)


def _read_tts_manifest(cfg: WwConfig) -> dict[str, list[Path]]:
    manifest = cfg.data.tts_out / "manifest.csv"
    groups: dict[str, list[Path]] = {"train": [], "val": [], "adversarial": []}
    with open(manifest) as f:
        for row in csv.DictReader(f):
            path = cfg.data.tts_out / row["path"]
            if row["label"] == "adversarial":
                groups["adversarial"].append(path)
            else:
                groups[row["split"]].append(path)
    return groups


def _read_recordings(cfg: WwConfig) -> list[Path]:
    split_manifest = cfg.data.recordings_root / "manifest_split.csv"
    if not split_manifest.exists():
        return []
    with open(split_manifest) as f:
        return [
            cfg.data.recordings_root / row["path"]
            for row in csv.DictReader(f)
            if row["split"] == "train"
        ]


def _local_negative_windows(frontend: WwFrontend, dirs: list[Path]) -> np.ndarray:
    ws = frontend.cfg.window_samples
    hop = int(LOCAL_NEG_HOP_S * 16000)
    feats = []
    for wav in scan_wavs(dirs):
        audio = load_mono(wav)
        for start in range(0, max(1, len(audio) - ws + 1), hop):
            chunk = audio[start : start + ws]
            if len(chunk) < ws:
                break
            feats.append(frontend.features(chunk * INT16_SCALE))
    if not feats:
        return np.zeros((0, frontend.cfg.head_frames, frontend.cfg.emb_dim), dtype=np.float16)
    return np.stack(feats).astype(np.float16)


def precompute(cfg: WwConfig, frontend: WwFrontend | None = None) -> Path:
    if frontend is None:
        frontend = WwFrontend.from_onnx(cfg.data.frontends_dir, DEFAULT_FRONTEND)
    rng = np.random.default_rng(cfg.training.seed)
    noise_paths = scan_wavs(cfg.augment.noise_dirs)
    rir_paths = scan_wavs([cfg.augment.rir_dir]) if cfg.augment.rir_dir.exists() else []
    groups = _read_tts_manifest(cfg)

    def tts_feats(paths: list[Path], augment: bool) -> list[np.ndarray]:
        out = []
        for path in paths:
            clip = load_mono(path)
            if augment:
                clip = augment_clip(clip, rng, cfg.augment, noise_paths, rir_paths)
            out.append(clip_features(frontend, clip))
        return out

    positives = tts_feats(groups["train"], augment=True)
    for path in _read_recordings(cfg):
        clip = load_mono(path)
        for _ in range(cfg.augment.augment_rounds_user):
            variant = augment_user_clip(clip, rng, cfg.augment, noise_paths, rir_paths)
            positives.append(clip_features(frontend, variant))

    cache = cfg.training.features_cache
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {
        "positives": np.concatenate(positives).astype(np.float16),
        "positives_val": np.concatenate(tts_feats(groups["val"], augment=False)).astype(np.float16),
        "negatives_adv": np.concatenate(tts_feats(groups["adversarial"], augment=True)).astype(
            np.float16
        ),
        "negatives_local": _local_negative_windows(frontend, cfg.data.negative_audio_dirs),
    }
    for name, arr in arrays.items():
        np.save(cache / f"{name}.npy", arr)
        print(f"{name}: {arr.shape}")
    return cache


class FeaturePools:
    """Batch sampler over the cached feature arrays + memory-mapped ACAV rows."""

    def __init__(self, cfg: WwConfig, seed: int):
        cache = cfg.training.features_cache
        self.rng = np.random.default_rng(seed)
        self.pos = np.load(cache / "positives.npy")
        self.adv = np.load(cache / "negatives_adv.npy")
        self.local = np.load(cache / "negatives_local.npy")
        if cfg.data.acav_features.exists():
            acav = np.load(cfg.data.acav_features, mmap_mode="r")
            n_train = max(0, len(acav) - ACAV_VAL_ROWS)
            self.acav = acav[: min(n_train, cfg.data.acav_subsample)]
            self.acav_val = acav[n_train:]
        else:
            print("warning: ACAV features missing — training negatives are adversarial/local only")
            self.acav = np.zeros((0,) + self.pos.shape[1:], dtype=np.float16)
            self.acav_val = self.acav

    def _draw(self, pool: np.ndarray, n: int) -> np.ndarray:
        if len(pool) == 0 or n == 0:
            return np.zeros((0,) + self.pos.shape[1:], dtype=np.float16)
        idx = np.sort(self.rng.integers(len(pool), size=n))  # sorted: kind to the mmap
        return np.asarray(pool[idx])

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_pos = batch_size // 4
        n_neg = batch_size - n_pos
        n_adv = n_neg // 4
        n_local = min(len(self.local), n_neg // 4)
        n_acav = n_neg - n_adv - n_local
        feats = np.concatenate(
            [
                self._draw(self.pos, n_pos),
                self._draw(self.adv, n_adv),
                self._draw(self.local, n_local),
                self._draw(self.acav, n_acav),
            ]
        ).astype(np.float32)
        labels = np.zeros(len(feats), dtype=np.float32)
        labels[:n_pos] = 1.0
        return torch.from_numpy(feats), torch.from_numpy(labels)


@torch.no_grad()
def _predict(model: WwHead, feats: torch.Tensor, device: torch.device) -> torch.Tensor:
    out = []
    for i in range(0, len(feats), 4096):
        out.append(model(feats[i : i + 4096].to(device)).cpu())
    return torch.cat(out) if out else torch.zeros(0)


def train(cfg: WwConfig) -> Path:
    t = cfg.training
    torch.manual_seed(t.seed)
    device = pick_device()
    print(f"device: {device}")

    pools = FeaturePools(cfg, t.seed)
    val_pos = torch.from_numpy(np.load(t.features_cache / "positives_val.npy").astype(np.float32))
    val_neg = torch.from_numpy(np.asarray(pools.acav_val[:20_000]).astype(np.float32))

    model = WwHead(t.layer_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t.steps)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    t.runs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(t.runs_dir)
    best_path = t.runs_dir / "best.pt"
    best_score = -1.0

    for step in range(1, t.steps + 1):
        model.train()
        feats, labels = pools.batch(t.batch_size)
        feats, labels = feats.to(device), labels.to(device)
        neg_weight = 1.0 + (t.max_negative_weight - 1.0) * (step / t.steps)
        weights = torch.where(
            labels > 0.5, torch.ones_like(labels), torch.full_like(labels, neg_weight)
        )
        loss = (criterion(model.logits(feats), labels) * weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        writer.add_scalar("loss/train", loss.item(), step)

        if step % t.val_every == 0 or step == t.steps:
            model.eval()
            recall = (_predict(model, val_pos, device) >= 0.5).float().mean().item()
            fp = 0.0
            if len(val_neg):
                fp = (_predict(model, val_neg, device) >= 0.5).float().mean().item()
            score = recall - fp
            writer.add_scalar("val/recall", recall, step)
            writer.add_scalar("val/fp_rate", fp, step)
            print(f"step {step}: loss {loss.item():.4f} recall {recall:.3f} fp {fp:.5f}")
            if score > best_score:
                best_score = score
                torch.save(
                    {"state_dict": model.state_dict(), "layer_size": t.layer_size}, best_path
                )

    writer.close()
    print(f"best: {best_path} (score {best_score:.4f})")
    return best_path


def load_head(checkpoint: Path) -> WwHead:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = WwHead(ckpt["layer_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["precompute", "train"])
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v1.yaml"))
    args = parser.parse_args()
    cfg = load_ww_config(args.config)
    if args.command == "precompute":
        precompute(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
