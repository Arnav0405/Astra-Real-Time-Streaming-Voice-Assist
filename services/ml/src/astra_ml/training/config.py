"""YAML-backed config shared by generation, training, and evaluation scripts."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    sources_root: Path
    chime_root: Path
    chime_prepared: Path
    libriparty_out: Path
    recipe_dir: Path

    @property
    def metadata_dir(self) -> Path:
        return self.libriparty_out / "metadata"

    def metadata_json(self, split: str) -> Path:
        """Locate the generated metadata JSON for a split, tolerant to naming."""
        candidates = [p for p in self.metadata_dir.glob("*.json") if split in p.name.lower()]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected one metadata json for {split!r} in {self.metadata_dir}, "
                f"found {[p.name for p in candidates]}"
            )
        return candidates[0]

    def mixtures_dir(self, split: str) -> Path:
        return self.libriparty_out / split


@dataclass
class TrainingConfig:
    crop_frames: int
    batch_size: int
    lr: float
    epochs: int
    patience: int
    grad_clip: float
    num_workers: int
    seed: int
    runs_dir: Path


@dataclass
class AugmentConfig:
    chime_noise_prob: float = 0.0
    snr_db_range: tuple[float, float] = (0.0, 20.0)
    negatives_fraction: float = 0.0


@dataclass
class Config:
    data: DataConfig
    training: TrainingConfig
    augment: AugmentConfig


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    data = {k: Path(v) for k, v in raw["data"].items()}
    t = raw["training"]
    t["runs_dir"] = Path(t["runs_dir"])
    a = raw.get("augment", {})
    if "snr_db_range" in a:
        a["snr_db_range"] = tuple(a["snr_db_range"])
    return Config(data=DataConfig(**data), training=TrainingConfig(**t), augment=AugmentConfig(**a))
