"""YAML-backed config for the wake word v2 (BC-ResNet) pipeline — configs/ww_v2.yaml.

Deliberately self-contained rather than importing from ww_config: v1 is scheduled for
deletion once v2 clears its gates, and the ~20 lines of overlapping dataclasses are a
cheaper debt than a cross-version import that has to be untangled then.

Data *generation* (piper TTS positives and adversarials, recording sessions) still runs
off configs/ww_v1.yaml — v2 reuses the clips it produced as-is and only needs to know
where they landed.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from astra_ml.audio.dft_mel import (
    KWS_HOP_SAMPLES,
    KWS_N_MELS,
    KWS_WIN_SAMPLES,
    KWS_WINDOW_SAMPLES,
)


@dataclass
class KwsDataConfig:
    tts_out: Path  # holds manifest.csv from ww_generate: positives + adversarials
    recordings_root: Path
    negative_recordings_root: Path
    # Environmental negatives (ESC-50, chime backgrounds), fold-split as in v1.
    negative_audio_dirs: list[Path] = field(default_factory=list)
    # Connected-speech negatives. v1 had none of these — its negative pool was ambient
    # only, which is why fa_per_hour_speech measured 345 against a gate of 5.
    speech_negative_dirs: list[Path] = field(default_factory=list)
    speech_negative_val_dirs: list[Path] = field(default_factory=list)
    frozen_test_sessions: list[str] = field(default_factory=list)
    frozen_negative_test_sessions: list[str] = field(default_factory=list)
    negative_folds: list[int] = field(default_factory=list)
    negative_val_folds: list[int] = field(default_factory=list)


@dataclass
class KwsFrontendConfig:
    window_samples: int = KWS_WINDOW_SAMPLES
    win_samples: int = KWS_WIN_SAMPLES
    hop_samples: int = KWS_HOP_SAMPLES
    n_mels: int = KWS_N_MELS
    chunk_samples: int = 1280  # 80 ms scoring cadence; window must be a multiple


@dataclass
class KwsModelConfig:
    # BC-ResNet-tau with base_c = 8 * tau. 24 = BC-ResNet-3, 54.2k params.
    base_c: int = 24
    sub_bands: int = 5


@dataclass
class KwsAugmentConfig:
    """Field names through user_speed_range match WwAugmentConfig so the existing
    astra_ml.data.ww_augment helpers accept this object unchanged."""

    rir_dir: Path
    noise_dirs: list[Path]
    rir_prob: float = 0.5
    noise_prob: float = 0.8  # the paper's background-noise probability
    snr_db_range: tuple[float, float] = (0.2, 15.0)
    user_pitch_semitones: tuple[float, float] = (-2.5, 2.0)
    user_speed_range: tuple[float, float] = (0.8, 1.1)
    # Keyword onset jitter around centre. 200 ms each way spans 400 ms = 5 consecutive
    # 80 ms scoring steps, so postproc's patience_frames=2 is satisfied with margin.
    positive_jitter_ms: float = 200.0
    # Windows holding less than this fraction of the keyword are emitted as label 0,
    # which sharpens the score onset instead of leaving the edges undefined.
    partial_max_frac: float = 0.6
    partial_negative_frac: float = 0.15
    # SpecAugment, no time warping. F follows tau: BC-ResNet-3 -> 5, with T fixed at 20.
    freq_mask_param: int = 5
    time_mask_param: int = 20
    n_freq_masks: int = 2
    n_time_masks: int = 2


@dataclass
class KwsTrainingConfig:
    runs_dir: Path
    seed: int = 1234
    # The paper's recipe, translated from 200 epochs x 36923 clips / batch 100 to steps.
    batch_size: int = 100
    steps: int = 75_000
    lr_peak: float = 0.1
    warmup_steps: int = 1_875  # 5/200 of training, as in the paper's 5 warmup epochs
    momentum: float = 0.9
    weight_decay: float = 1e-3
    val_every: int = 500
    val_batches: int = 40
    num_workers: int = 4
    # Share of each batch that is positive, and the share of those drawn from real user
    # recordings rather than TTS. 0.25 for the same reason as v1: the recordings are one
    # speaker in a handful of rooms, and a larger share fits a speaker detector.
    positive_frac: float = 0.25
    user_positive_frac: float = 0.25


@dataclass
class KwsPostprocDefaults:
    threshold: float
    patience_frames: int
    refractory_seconds: float


@dataclass
class KwsGatingConfig:
    preroll_frames: int
    partial_chunk: str


@dataclass
class KwsEvalConfig:
    fa_audio_dirs: list[Path]
    recall_floor_quiet: float
    recall_floor_noisy: float
    max_fa_per_hour: float
    max_latency_ms: float
    fa_folds: list[int] = field(default_factory=list)
    fa_speech_manifests: list[Path] = field(default_factory=list)
    # Speech FA is gated apart from fa_audio_dirs so hours of easy ambient audio cannot
    # dilute the one number that measures "fires on any word I say".
    fa_speech_dirs: list[Path] = field(default_factory=list)
    max_fa_per_hour_speech: float = 5.0


@dataclass
class KwsConfig:
    data: KwsDataConfig
    augment: KwsAugmentConfig
    training: KwsTrainingConfig
    postproc: KwsPostprocDefaults
    gating: KwsGatingConfig
    eval: KwsEvalConfig
    frontend: KwsFrontendConfig = field(default_factory=KwsFrontendConfig)
    model: KwsModelConfig = field(default_factory=KwsModelConfig)


_PATH_KEYS = {"tts_out", "recordings_root", "negative_recordings_root", "rir_dir", "runs_dir"}
_PATH_LIST_KEYS = {
    "negative_audio_dirs",
    "speech_negative_dirs",
    "speech_negative_val_dirs",
    "noise_dirs",
    "fa_audio_dirs",
    "fa_speech_dirs",
    "fa_speech_manifests",
}
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


def _check_folds_disjoint(data: KwsDataConfig, eval_cfg: KwsEvalConfig) -> None:
    """A fold in two roles silently restores the leak the split exists to remove."""
    named = {
        "negative_folds": set(data.negative_folds),
        "negative_val_folds": set(data.negative_val_folds),
        "fa_folds": set(eval_cfg.fa_folds),
    }
    for a, b in (
        ("negative_folds", "negative_val_folds"),
        ("negative_folds", "fa_folds"),
        ("negative_val_folds", "fa_folds"),
    ):
        overlap = named[a] & named[b]
        if overlap:
            raise ValueError(f"{a} and {b} share fold(s) {sorted(overlap)}; they must be disjoint")


def _check_window(frontend: KwsFrontendConfig) -> None:
    """The Go detector advances the window one chunk at a time (detector.go:36-37), and
    the mel framing must land exactly on the window edge."""
    if frontend.window_samples % frontend.chunk_samples:
        raise ValueError(
            f"window_samples {frontend.window_samples} must be a multiple of "
            f"chunk_samples {frontend.chunk_samples}"
        )
    if (frontend.window_samples - frontend.win_samples) % frontend.hop_samples:
        raise ValueError(
            f"window_samples {frontend.window_samples} does not land on a whole mel frame "
            f"for win {frontend.win_samples} / hop {frontend.hop_samples}"
        )
    if frontend.n_mels != KWS_N_MELS:
        raise ValueError(
            f"n_mels must be {KWS_N_MELS}: BC-ResNet's classifier collapses freq 5 -> 1 with a "
            f"kernel-5 conv, and SubSpectralNorm needs freq divisible by 5"
        )


def load_kws_config(path: Path) -> KwsConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = KwsConfig(
        data=KwsDataConfig(**_convert(raw["data"])),
        augment=KwsAugmentConfig(**_convert(raw["augment"])),
        training=KwsTrainingConfig(**_convert(raw["training"])),
        postproc=KwsPostprocDefaults(**raw["postproc"]),
        gating=KwsGatingConfig(**raw["gating"]),
        eval=KwsEvalConfig(**_convert(raw["eval"])),
        frontend=KwsFrontendConfig(**raw.get("frontend", {})),
        model=KwsModelConfig(**raw.get("model", {})),
    )
    _check_folds_disjoint(cfg.data, cfg.eval)
    _check_window(cfg.frontend)
    return cfg
