"""On-the-fly window construction for wake-word v2 (BC-ResNet) training.

v1 had to precompute features to disk because its frontend was a frozen ONNX graph:
augmentation could only happen before featurization, so it was baked into a fixed number
of rounds. v2 trains the frontend's consumer directly, so augmentation moves into the
DataLoader and every epoch sees fresh RIRs, noise, levels and alignments. Caching is not
an option at this scale anyway — 4.5M negative windows at 15360 int16 samples is ~138 GB.

Batch composition is positional rather than sampled: __getitem__ derives its role from
`index % batch_size`, so a plain DataLoader with shuffle=False yields exactly
positive_frac positives and evenly quartered negatives in every batch, with no custom
sampler. The index also seeds the per-item RNG, so runs are reproducible.

Windows are returned as float32 in raw int16 range (±32768), the scale the exported graph
and the Go runtime agree on (engine.go:51-53).
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from astra_ml.data.ww_augment import (
    apply_rir,
    load_mono,
    pitch_shift,
    scan_wavs,
    scan_wavs_in_folds,
    speed_perturb,
)
from astra_ml.training.kws_config import KwsAugmentConfig, KwsConfig

SR = 16000
INT16_SCALE = 32768.0
AUDIO_EXTS = (".wav", ".flac")
# Level randomization: log-mel is not level invariant (a gain is an additive offset the
# per-sample statistics cannot absorb), and the mic's absolute level varies with distance.
PEAK_RANGE = (0.15, 0.95)
QUIET_BACKGROUND_SNR_DB = 30.0
TRIM_TOP_DB = 35.0


# --- source scanning -------------------------------------------------------------


def scan_audio(dirs: list[Path], exts: tuple[str, ...] = AUDIO_EXTS) -> list[Path]:
    """scan_wavs, but for corpora that ship flac (LibriSpeech)."""
    return sorted(p for d in dirs for e in exts for p in Path(d).rglob(f"*{e}"))


def _read_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def tts_clips(tts_out: Path, label: str, split: str) -> list[Path]:
    rows = _read_manifest(tts_out / "manifest.csv")
    return [tts_out / r["path"] for r in rows if r["label"] == label and r["split"] == split]


def recording_clips(root: Path, splits: tuple[str, ...]) -> list[Path]:
    manifest = root / "manifest_split.csv"
    if not manifest.exists():
        return []
    return [root / r["path"] for r in _read_manifest(manifest) if r["split"] in splits]


# --- signal helpers --------------------------------------------------------------


def trim_silence(clip: np.ndarray, frame: int = 320, top_db: float = TRIM_TOP_DB) -> np.ndarray:
    """Crop to the spoken span. TTS clips carry leading and trailing silence, and both
    the ±jitter and the partial-window fraction are meaningless measured against it."""
    n = clip.size // frame * frame
    if n == 0:
        return clip
    rms = np.sqrt(np.square(clip[:n].reshape(-1, frame)).mean(axis=1))
    peak = float(rms.max())
    if peak <= 0:
        return clip
    keep = np.nonzero(rms >= peak * 10 ** (-top_db / 20))[0]
    if keep.size == 0:
        return clip
    return clip[keep[0] * frame : (keep[-1] + 1) * frame]


def mix_at_snr(
    signal: np.ndarray, region: np.ndarray, background: np.ndarray, snr_db: float
) -> np.ndarray:
    """Add `background` under `signal` at `snr_db` measured against `region`.

    ww_augment.add_noise measures SNR over the whole clip, which is wrong here: most of a
    positive window is padding, so a window-wide RMS would make every background quieter
    than requested by however much silence surrounds the keyword.
    """
    sig_power = float(np.square(region).mean())
    bg_power = float(np.square(background).mean())
    if sig_power <= 0 or bg_power <= 0:
        return signal
    gain = np.sqrt(sig_power / (bg_power * 10 ** (snr_db / 10)))
    return (signal + np.float32(gain) * background).astype(np.float32)


def random_gain(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    peak = float(np.abs(clip).max())
    if peak <= 0:
        return clip
    return (clip * (rng.uniform(*PEAK_RANGE) / peak)).astype(np.float32)


def _tile_to(clip: np.ndarray, n: int) -> np.ndarray:
    if clip.size == 0:
        return np.zeros(n, dtype=np.float32)
    if clip.size < n:
        clip = np.tile(clip, -(-n // clip.size))
    return clip[:n]


def random_crop(path: Path, rng: np.random.Generator, n: int) -> np.ndarray:
    """n samples at 16 kHz from a random offset in `path`.

    16 kHz sources (LibriSpeech flac, chime backgrounds) are read windowed, so a 35 s
    utterance costs one seek rather than a full decode. Anything else (ESC-50 is 44.1 kHz)
    is decoded whole and resampled, which is what load_mono already does.
    """
    info = sf.info(path)
    if info.samplerate == SR and info.frames > n:
        start = int(rng.integers(0, info.frames - n))
        audio, _ = sf.read(path, start=start, frames=n, dtype="float32", always_2d=True)
        return np.ascontiguousarray(audio[:, 0])
    audio = load_mono(path)
    if audio.size > n:
        start = int(rng.integers(0, audio.size - n))
        return audio[start : start + n]
    return _tile_to(audio, n)


# --- window construction ---------------------------------------------------------


@dataclass
class Placement:
    """Where a trimmed keyword sits in the window, and how much of it made it in."""

    start: int  # index into the window; the clip is sliced to fit
    clip_start: int
    clip_stop: int

    @property
    def n_inside(self) -> int:
        return self.clip_stop - self.clip_start


def place_full(clip_len: int, window: int, jitter: int, rng: np.random.Generator) -> Placement:
    """Whole keyword inside the window, centred ± jitter."""
    if clip_len >= window:
        offset = int(rng.integers(0, clip_len - window + 1))
        return Placement(start=0, clip_start=offset, clip_stop=offset + window)
    centre = (window - clip_len) // 2
    lo, hi = max(0, centre - jitter), min(window - clip_len, centre + jitter)
    return Placement(start=int(rng.integers(lo, hi + 1)), clip_start=0, clip_stop=clip_len)


def place_partial(
    clip_len: int, window: int, max_frac: float, rng: np.random.Generator
) -> Placement:
    """Keyword hanging off one edge, with between 10% and max_frac of it inside.

    These are labelled 0. Without them the score has no defined behaviour on the way in
    and out of the keyword, and a blurred onset both weakens postproc's patience gate and
    smears the frame index the endpoint machine arms on (machine.go:67-71).
    """
    inside = int(clip_len * rng.uniform(0.1, max_frac))
    inside = max(1, min(inside, window, clip_len))
    if rng.random() < 0.5:  # tail of the keyword at the window start
        return Placement(start=0, clip_start=clip_len - inside, clip_stop=clip_len)
    return Placement(start=window - inside, clip_start=0, clip_stop=inside)


class KwsWindowBuilder:
    """Turns a source clip plus a background pool into one training window."""

    def __init__(self, cfg: KwsAugmentConfig, window_samples: int) -> None:
        self.cfg = cfg
        self.window = window_samples
        self.jitter = int(cfg.positive_jitter_ms * SR / 1000)
        self.noise_paths = scan_wavs(cfg.noise_dirs) if cfg.noise_dirs else []
        self.rir_paths = scan_wavs([cfg.rir_dir]) if cfg.rir_dir else []

    def background(self, rng: np.random.Generator) -> np.ndarray:
        if not self.noise_paths:
            return np.zeros(self.window, dtype=np.float32)
        path = self.noise_paths[int(rng.integers(len(self.noise_paths)))]
        return random_crop(path, rng, self.window)

    def _reverb(self, clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.rir_paths and rng.random() < self.cfg.rir_prob:
            rir = load_mono(self.rir_paths[int(rng.integers(len(self.rir_paths)))])
            return apply_rir(clip, rir)
        return clip

    def perturb_user(self, clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Pitch/speed jitter, applied only to real recordings: ~190 clips from one
        speaker in a handful of rooms otherwise trains a speaker detector."""
        clip = pitch_shift(clip, rng.uniform(*self.cfg.user_pitch_semitones))
        return speed_perturb(clip, rng.uniform(*self.cfg.user_speed_range))

    def _snr(self, rng: np.random.Generator) -> float:
        if rng.random() < self.cfg.noise_prob:
            return float(rng.uniform(*self.cfg.snr_db_range))
        # Still a real background rather than digital silence: the runtime only ever
        # scores VAD-gated audio with 1 s of real preroll behind it (sink.go:163-172),
        # so a window padded with zeros is a window shape the model never meets live.
        return QUIET_BACKGROUND_SNR_DB

    def from_clip(
        self, clip: np.ndarray, rng: np.random.Generator, partial: bool = False
    ) -> np.ndarray:
        """Place a short clip (keyword, adversarial phrase, recorded utterance) in a
        background window."""
        clip = self._reverb(trim_silence(clip), rng)
        if clip.size == 0:
            return self.background(rng)
        if partial:
            p = place_partial(clip.size, self.window, self.cfg.partial_max_frac, rng)
        else:
            p = place_full(clip.size, self.window, self.jitter, rng)

        signal = np.zeros(self.window, dtype=np.float32)
        piece = clip[p.clip_start : p.clip_stop]
        signal[p.start : p.start + piece.size] = piece
        mixed = mix_at_snr(signal, piece, self.background(rng), self._snr(rng))
        return random_gain(mixed, rng)

    def from_stream(self, path: Path, rng: np.random.Generator) -> np.ndarray:
        """Crop a window straight out of a long recording (LibriSpeech, chime, ESC-50)."""
        window = self._reverb(random_crop(path, rng, self.window), rng)
        if rng.random() < self.cfg.noise_prob:
            snr = float(rng.uniform(*self.cfg.snr_db_range))
            window = mix_at_snr(window, window, self.background(rng), snr)
        return random_gain(window, rng)


# --- pools and dataset -----------------------------------------------------------


@dataclass
class KwsPools:
    """Clip-like and stream-like sources, already split."""

    positives_tts: list[Path]
    positives_user: list[Path]
    adversarial: list[Path]
    speech: list[Path]  # LibriSpeech — stream-like
    environment: list[Path]  # ESC-50 / chime — stream-like
    negatives_user: list[Path]

    def check(self) -> None:
        empty = [name for name, v in vars(self).items() if not v]
        if empty:
            raise ValueError(f"empty source pool(s): {empty}")


def build_pools(cfg: KwsConfig, split: str = "train") -> KwsPools:
    data = cfg.data
    if split == "train":
        return KwsPools(
            positives_tts=tts_clips(data.tts_out, "positive", "train"),
            positives_user=recording_clips(data.recordings_root, ("train",)),
            adversarial=tts_clips(data.tts_out, "adversarial", "train"),
            speech=scan_audio(data.speech_negative_dirs),
            environment=scan_wavs_in_folds(data.negative_audio_dirs, data.negative_folds),
            negatives_user=recording_clips(data.negative_recordings_root, ("train",)),
        )
    return KwsPools(
        positives_tts=tts_clips(data.tts_out, "positive", "val"),
        positives_user=recording_clips(data.recordings_root, ("eval",)),
        adversarial=tts_clips(data.tts_out, "adversarial", "train"),
        speech=scan_audio(data.speech_negative_val_dirs),
        environment=scan_wavs_in_folds(data.negative_audio_dirs, data.negative_val_folds),
        negatives_user=recording_clips(data.negative_recordings_root, ("eval",)),
    )


class KwsWindows(Dataset):
    """Fixed-composition stream of (window[float32, W], label[float32]).

    Slot layout within a batch of B, for positive_frac p:
        slots [0, pB)        positives, of which user_positive_frac come from recordings
                             and partial_negative_frac are re-labelled 0
        slots [pB, B)        negatives, cycling: speech, adversarial, environment, user
    """

    NEGATIVE_ORDER = ("speech", "adversarial", "environment", "negatives_user")

    def __init__(
        self,
        cfg: KwsConfig,
        pools: KwsPools,
        length: int,
        seed: int | None = None,
    ) -> None:
        self.pools = pools
        self.augment = cfg.augment
        self.batch_size = cfg.training.batch_size
        self.n_positive = max(1, round(cfg.training.positive_frac * self.batch_size))
        self.user_positive_frac = cfg.training.user_positive_frac
        self.seed = cfg.training.seed if seed is None else seed
        self.length = length
        self.builder = KwsWindowBuilder(cfg.augment, cfg.frontend.window_samples)

    def __len__(self) -> int:
        return self.length

    def _pick(self, paths: list[Path], rng: np.random.Generator) -> Path:
        return paths[int(rng.integers(len(paths)))]

    def _positive(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        user = bool(rng.random() < self.user_positive_frac) and bool(self.pools.positives_user)
        pool = self.pools.positives_user if user else self.pools.positives_tts
        clip = load_mono(self._pick(pool, rng))
        if user:
            clip = self.builder.perturb_user(clip, rng)
        partial = rng.random() < self.augment.partial_negative_frac
        return self.builder.from_clip(clip, rng, partial=partial), 0.0 if partial else 1.0

    def _negative(self, slot: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        kind = self.NEGATIVE_ORDER[slot % len(self.NEGATIVE_ORDER)]
        paths = getattr(self.pools, kind)
        path = self._pick(paths, rng)
        if kind in ("speech", "environment"):
            return self.builder.from_stream(path, rng), 0.0
        clip = load_mono(path)
        if kind == "negatives_user":
            clip = self.builder.perturb_user(clip, rng)
        return self.builder.from_clip(clip, rng), 0.0

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float32]:
        rng = np.random.default_rng([self.seed, index])
        slot = index % self.batch_size
        if slot < self.n_positive:
            window, label = self._positive(rng)
        else:
            window, label = self._negative(slot - self.n_positive, rng)
        return (window * INT16_SCALE).astype(np.float32), np.float32(label)
