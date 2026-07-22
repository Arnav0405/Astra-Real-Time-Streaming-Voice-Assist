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
from astra_ml.data.oww_assets import load_acav
from astra_ml.data.ww_augment import (
    augment_clip,
    augment_user_clip,
    load_mono,
    scan_wavs,
    scan_wavs_in_folds,
)
from astra_ml.models.ww import WwHead
from astra_ml.training.ww_config import WwConfig, load_ww_config

INT16_SCALE = 32767.0  # frontend consumes int16-range floats
JITTER_OFFSETS = (0, 800, 1600)  # word end at window end, -50 ms, -100 ms
ACAV_VAL_ROWS = 100_000
SCORE_STEPS_PER_HOUR = 16000 * 3600 / 1280  # 45k 80 ms chunks; must match ww_eval cadence
# Matches ww_eval's 1280-sample (80 ms) scoring cadence exactly. An earlier 0.16 s compromise
# still left every other eval window at an untrained alignment, and false accepts stayed high
# (59/hr), so the pool now covers every offset the model is graded on.
LOCAL_NEG_HOP_S = 0.08
# Checkpoint-selection FPR. The shipping gate (~1.3e-5) over ~1e5 val windows puts the
# quantile between the top two negative scores, so "best" was decided by noise.
SELECT_FPR = 1e-3


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


def _local_negative_windows(
    frontend: WwFrontend, dirs: list[Path], folds: list[int] | None = None
) -> np.ndarray:
    ws = frontend.cfg.window_samples
    hop = int(LOCAL_NEG_HOP_S * 16000)
    feats = []
    for wav in scan_wavs_in_folds(dirs, folds or []):
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
    # same folds as the negatives: any audio training touches, in any role, must stay out of
    # eval.fa_folds. Augmentation noise counts — it is mixed under every positive.
    noise_paths = scan_wavs_in_folds(cfg.augment.noise_dirs, cfg.data.negative_folds)
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
    # kept in its own array, not concatenated into positives: merging them makes the TTS:user
    # ratio a property of the cache (20:1 here) instead of something training can control.
    user_positives = []
    for path in _read_recordings(cfg):
        clip = load_mono(path)
        for _ in range(cfg.augment.augment_rounds_user):
            variant = augment_user_clip(clip, rng, cfg.augment, noise_paths, rir_paths)
            user_positives.append(clip_features(frontend, variant))

    cache = cfg.training.features_cache
    cache.mkdir(parents=True, exist_ok=True)
    empty = np.zeros(
        (0, frontend.cfg.head_frames, frontend.cfg.emb_dim), dtype=np.float16
    )
    arrays = {
        "positives": np.concatenate(positives).astype(np.float16),
        "positives_user": (
            np.concatenate(user_positives).astype(np.float16) if user_positives else empty
        ),
        "positives_val": np.concatenate(tts_feats(groups["val"], augment=False)).astype(np.float16),
        "negatives_adv": np.concatenate(tts_feats(groups["adversarial"], augment=True)).astype(
            np.float16
        ),
        "negatives_local": _local_negative_windows(
            frontend, cfg.data.negative_audio_dirs, cfg.data.negative_folds
        ),
        # held-out folds of the same corpora ww_eval measures FA on, so val FP tracks the
        # gate instead of ACAV's much easier distribution (2.2 est fa/hr vs 59 measured)
        "negatives_local_val": _local_negative_windows(
            frontend, cfg.data.negative_audio_dirs, cfg.data.negative_val_folds
        ),
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
        user_path = cache / "positives_user.npy"
        self.user = np.load(user_path) if user_path.exists() else self.pos[:0]
        self.user_frac = cfg.training.user_positive_frac if len(self.user) else 0.0
        if not len(self.user):
            print("warning: no user positives — training on TTS voices alone")
        adv = np.load(cache / "negatives_adv.npy")
        # ponytail: random split, not phrase-contiguous — every phrase must stay in train
        perm = self.rng.permutation(len(adv))
        n_adv_val = len(adv) // 10
        self.adv, self.adv_val = adv[perm[n_adv_val:]], adv[perm[:n_adv_val]]
        self.local = np.load(cache / "negatives_local.npy")
        local_val_path = cache / "negatives_local_val.npy"
        self.local_val = np.load(local_val_path) if local_val_path.exists() else self.pos[:0]
        if cfg.data.acav_features.exists():
            acav = load_acav(cfg.data.acav_features)
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
        n_user = int(n_pos * self.user_frac)
        n_tts = n_pos - n_user
        n_adv = n_neg // 4
        n_local = min(len(self.local), n_neg // 4)
        n_acav = n_neg - n_adv - n_local
        feats = np.concatenate(
            [
                self._draw(self.pos, n_tts),
                self._draw(self.user, n_user),
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
    val_neg = torch.from_numpy(
        np.concatenate(
            [np.asarray(pools.acav_val), pools.adv_val, pools.local_val]
        ).astype(np.float32)
    )
    val_neg_fa = torch.from_numpy(np.asarray(pools.local_val).astype(np.float32))
    pp_threshold = cfg.postproc.threshold
    print(f"select fpr {SELECT_FPR:.0e}, val_neg {len(val_neg)}")
    print(f"fa-domain val pool: {len(val_neg_fa)} windows @ threshold {pp_threshold}")

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
        raw = criterion(model.logits(feats), labels)
        loss = (raw * weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        writer.add_scalar("loss/train", loss.item(), step)
        writer.add_scalar("loss/train_unweighted", raw.mean().item(), step)
        # the weight ramp grows the loss scale ~500x over the run; divide it out so the
        # curve is comparable across steps
        writer.add_scalar("loss/train_stationary", loss.item() / weights.mean().item(), step)

        if step % t.val_every == 0 or step == t.steps:
            model.eval()
            pos_scores = _predict(model, val_pos, device)
            neg_scores = _predict(model, val_neg, device)
            recall = (pos_scores >= 0.5).float().mean().item()
            fp = (neg_scores >= 0.5).float().mean().item() if len(neg_scores) else 0.0
            # FA estimated on held-out folds of the eval corpora; falls back to the mixed pool
            # when no folds are configured, where it stays optimistic.
            fa_pool = val_neg_fa if len(val_neg_fa) else val_neg
            fa_neg_scores = _predict(model, fa_pool, device)
            fa_fp = (
                (fa_neg_scores >= pp_threshold).float().mean().item() if len(fa_neg_scores) else 0.0
            )
            est_fa = fa_fp * SCORE_STEPS_PER_HOUR
            # Select on the FA-domain pool, not the ACAV-dominated val_neg: the old score
            # ignored the corpora the fa_per_hour gate measures, so est FA climbed
            # 5 -> 200/hr across training while "best" kept improving.
            if len(fa_neg_scores):
                thresh = torch.quantile(fa_neg_scores, 1.0 - SELECT_FPR).item()
                score = (pos_scores > thresh).float().mean().item()
            else:
                score = recall
            writer.add_scalar("val/recall", recall, step)
            writer.add_scalar("val/fp_rate", fp, step)
            writer.add_scalar("val/recall_at_select_fpr", score, step)
            # the only val number directly comparable to the eval.max_fa_per_hour gate
            writer.add_scalar("val/est_fa_per_hour", est_fa, step)
            print(
                f"step {step}: loss {loss.item():.4f} recall {recall:.3f} "
                f"fp {fp:.5f} (~{est_fa:.1f} fa/hr) recall@fpr {score:.3f}"
            )
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
