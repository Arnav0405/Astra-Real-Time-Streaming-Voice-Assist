"""Per-clip augmentation for wake-word training data.

Numpy-in, numpy-out (float32, 16 kHz, [-1, 1]); applied on the fly during
feature precompute, never materialized as wav files. TTS clips get RIR
convolution + additive noise; user recordings additionally get pitch/speed
perturbation (augment_user_clip) so ~100 real clips stretch into thousands of
training variants. Same SNR mixing formula as astra_ml.data.augment.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy.signal import fftconvolve

from astra_ml.training.ww_config import WwAugmentConfig

SR = 16000


def scan_wavs(dirs: list[Path]) -> list[Path]:
    return sorted(p for d in dirs for p in Path(d).rglob("*.wav"))


def load_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio[:, 0]
    if sr != SR:
        audio = soxr.resample(audio, sr, SR)
    return audio


def add_noise(clip: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(clip):
        noise = np.tile(noise, -(-len(clip) // len(noise)))
    noise = noise[: len(clip)]
    noise_power = float(np.square(noise).mean())
    if noise_power == 0:
        return clip
    gain = np.sqrt(float(np.square(clip).mean()) / (noise_power * 10 ** (snr_db / 10)))
    return clip + gain.astype(np.float32) * noise


def apply_rir(clip: np.ndarray, rir: np.ndarray) -> np.ndarray:
    rir = rir / (np.abs(rir).max() + 1e-9)
    return fftconvolve(clip, rir)[: len(clip)].astype(np.float32)


def pitch_shift(clip: np.ndarray, semitones: float) -> np.ndarray:
    import librosa

    return librosa.effects.pitch_shift(clip, sr=SR, n_steps=semitones)


def speed_perturb(clip: np.ndarray, rate: float) -> np.ndarray:
    """rate > 1 -> faster (shorter); pitch shifts with it, like sox speed."""
    return soxr.resample(clip, int(SR * rate), SR).astype(np.float32)


def _peak_norm(clip: np.ndarray) -> np.ndarray:
    peak = np.abs(clip).max()
    return clip / peak if peak > 1 else clip


def augment_clip(
    clip: np.ndarray,
    rng: np.random.Generator,
    cfg: WwAugmentConfig,
    noise_paths: list[Path],
    rir_paths: list[Path],
) -> np.ndarray:
    if rir_paths and rng.random() < cfg.rir_prob:
        clip = apply_rir(clip, load_mono(rir_paths[rng.integers(len(rir_paths))]))
    if noise_paths and rng.random() < cfg.noise_prob:
        snr = rng.uniform(*cfg.snr_db_range)
        clip = add_noise(clip, load_mono(noise_paths[rng.integers(len(noise_paths))]), snr)
    return _peak_norm(clip)


def augment_user_clip(
    clip: np.ndarray,
    rng: np.random.Generator,
    cfg: WwAugmentConfig,
    noise_paths: list[Path],
    rir_paths: list[Path],
) -> np.ndarray:
    clip = pitch_shift(clip, rng.uniform(*cfg.user_pitch_semitones))
    clip = speed_perturb(clip, rng.uniform(*cfg.user_speed_range))
    return augment_clip(clip, rng, cfg, noise_paths, rir_paths)
