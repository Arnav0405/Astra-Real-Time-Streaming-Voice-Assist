"""YAML-backed config for the wake word pipeline (configs/ww_v1.yaml)."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class WwDataConfig:
    frontends_dir: Path
    voices_dir: Path
    voices: list[str]
    tts_out: Path
    spellings: list[str]
    n_positives: int
    n_positives_val: int
    adversarial_phrases: list[str]
    n_adversarial_per_phrase: int
    acav_features: Path
    acav_subsample: int
    negative_audio_dirs: list[Path]
    recordings_root: Path
    frozen_test_sessions: list[str] = field(default_factory=list)
    family_sessions: list[str] = field(default_factory=list)

    def voice_paths(self) -> list[Path]:
        return [self.voices_dir / f"{name}.onnx" for name in self.voices]


@dataclass
class WwAugmentConfig:
    rir_dir: Path
    rir_prob: float
    noise_prob: float
    snr_db_range: tuple[float, float]
    noise_dirs: list[Path]
    user_pitch_semitones: tuple[float, float]
    user_speed_range: tuple[float, float]
    augment_rounds_user: int


@dataclass
class WwTrainingConfig:
    batch_size: int
    steps: int
    lr: float
    layer_size: int
    max_negative_weight: float
    val_every: int
    seed: int
    features_cache: Path
    runs_dir: Path
    # Fraction of each batch's positives drawn from real user recordings rather than TTS.
    # Decoupled from augment_rounds_user on purpose: the mix is a training knob, and tying it
    # to how many augmented copies sit on disk means re-running precompute to change it.
    user_positive_frac: float = 0.5


@dataclass
class WwPostprocDefaults:
    threshold: float
    patience_frames: int
    refractory_seconds: float


@dataclass
class WwGatingConfig:
    preroll_frames: int
    partial_chunk: str


@dataclass
class WwEvalConfig:
    fa_audio_dirs: list[Path]
    recall_floor_quiet: float
    recall_floor_noisy: float
    max_fa_per_hour: float
    max_latency_ms: float


@dataclass
class WwConfig:
    data: WwDataConfig
    augment: WwAugmentConfig
    training: WwTrainingConfig
    postproc: WwPostprocDefaults
    gating: WwGatingConfig
    eval: WwEvalConfig


_PATH_KEYS = {
    "frontends_dir",
    "voices_dir",
    "tts_out",
    "acav_features",
    "recordings_root",
    "rir_dir",
    "features_cache",
    "runs_dir",
}
_PATH_LIST_KEYS = {"negative_audio_dirs", "noise_dirs", "fa_audio_dirs"}
_TUPLE_KEYS = {"snr_db_range", "user_pitch_semitones", "user_speed_range"}


def _convert(section: dict) -> dict:
    out = {}
    for k, v in section.items():
        if k in _PATH_KEYS:
            out[k] = Path(v)
        elif k in _PATH_LIST_KEYS:
            out[k] = [Path(p) for p in v]
        elif k in _TUPLE_KEYS:
            out[k] = tuple(v)
        else:
            out[k] = v
    return out


def load_ww_config(path: Path) -> WwConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return WwConfig(
        data=WwDataConfig(**_convert(raw["data"])),
        augment=WwAugmentConfig(**_convert(raw["augment"])),
        training=WwTrainingConfig(**_convert(raw["training"])),
        postproc=WwPostprocDefaults(**raw["postproc"]),
        gating=WwGatingConfig(**raw["gating"]),
        eval=WwEvalConfig(**_convert(raw["eval"])),
    )
