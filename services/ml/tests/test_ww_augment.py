"""Tests for wake-word augmentation on synthetic tones."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from astra_ml.data import ww_augment
from astra_ml.data.ww_augment import (
    SR,
    add_noise,
    apply_rir,
    augment_clip,
    augment_user_clip,
    pitch_shift,
    scan_wavs,
    scan_wavs_in_folds,
    speed_perturb,
    wav_fold,
)
from astra_ml.training.ww_config import WwAugmentConfig


def tone(freq: float, seconds: float = 0.5, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(clip: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(clip))
    return float(np.fft.rfftfreq(len(clip), 1 / SR)[spectrum.argmax()])


def cfg_with(**overrides) -> WwAugmentConfig:
    base = dict(
        rir_dir="unused",
        rir_prob=1.0,
        noise_prob=1.0,
        snr_db_range=(10.0, 10.0),
        noise_dirs=[],
        user_pitch_semitones=(2.0, 2.0),
        user_speed_range=(1.1, 1.1),
        augment_rounds_user=2,
    )
    base.update(overrides)
    return WwAugmentConfig(**base)


def test_add_noise_snr_direction():
    clip, noise = tone(440), np.random.default_rng(0).normal(0, 0.1, SR // 2).astype(np.float32)
    quiet = add_noise(clip, noise, snr_db=20.0)
    loud = add_noise(clip, noise, snr_db=0.0)
    assert np.square(loud - clip).mean() > np.square(quiet - clip).mean()


def test_add_noise_tiles_short_noise():
    clip = tone(440, seconds=1.0)
    noise = np.full(100, 0.05, dtype=np.float32)
    out = add_noise(clip, noise, snr_db=10.0)
    assert out.shape == clip.shape
    assert not np.allclose(out, clip)


def test_apply_rir_preserves_length_and_changes_signal():
    clip = tone(440)
    rir = np.zeros(200, dtype=np.float32)
    rir[0], rir[150] = 1.0, 0.6  # direct path + one echo
    out = apply_rir(clip, rir)
    assert out.shape == clip.shape
    assert not np.allclose(out, clip)


def test_pitch_shift_moves_dominant_frequency():
    shifted = pitch_shift(tone(440, seconds=1.0), semitones=12.0)
    assert dominant_freq(shifted) == pytest.approx(880, rel=0.05)


def test_speed_perturb_changes_length():
    clip = tone(440, seconds=1.0)
    faster = speed_perturb(clip, 1.25)
    assert len(faster) == pytest.approx(len(clip) / 1.25, rel=0.01)


def test_augment_clip_deterministic_and_normalized(tmp_path):
    noise_path = tmp_path / "noise.wav"
    sf.write(noise_path, np.random.default_rng(1).normal(0, 0.2, SR).astype(np.float32), SR)
    rir_path = tmp_path / "rir.wav"
    rir = np.zeros(100, dtype=np.float32)
    rir[0] = 1.0
    sf.write(rir_path, rir, SR)

    clip = tone(440, amp=0.9)
    cfg = cfg_with(snr_db_range=(0.0, 0.0))
    out1 = augment_clip(clip, np.random.default_rng(7), cfg, [noise_path], [rir_path])
    out2 = augment_clip(clip, np.random.default_rng(7), cfg, [noise_path], [rir_path])
    np.testing.assert_array_equal(out1, out2)
    assert np.abs(out1).max() <= 1.0


def test_probability_zero_is_identity():
    clip = tone(440)
    cfg = cfg_with(rir_prob=0.0, noise_prob=0.0)
    out = augment_clip(clip, np.random.default_rng(0), cfg, [], [])
    np.testing.assert_array_equal(out, clip)


def test_augment_user_clip_applies_pitch_and_speed():
    clip = tone(440, seconds=1.0)
    cfg = cfg_with(rir_prob=0.0, noise_prob=0.0)
    out = augment_user_clip(clip, np.random.default_rng(0), cfg, [], [])
    assert len(out) == pytest.approx(len(clip) / 1.1, rel=0.02)
    assert dominant_freq(out) == pytest.approx(440 * 2 ** (2 / 12) * 1.1, rel=0.05)


def test_scan_wavs_recursive_sorted(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("b.wav", "a.wav", "sub/c.wav"):
        sf.write(tmp_path / name, np.zeros(100, dtype=np.float32), SR)
    got = scan_wavs([tmp_path])
    assert [p.name for p in got] == ["a.wav", "b.wav", "c.wav"]


def test_load_mono_resamples(tmp_path):
    path = tmp_path / "x.wav"
    sf.write(path, np.zeros(22050, dtype=np.float32), 22050)
    audio = ww_augment.load_mono(path)
    assert len(audio) == SR


def test_wav_fold_reads_esc50_naming():
    # ESC-50 encodes its official fold as the leading digit of the filename
    assert wav_fold(Path("4-100032-A-0.wav"), index=0) == 4
    assert wav_fold(Path("1-100038-A-14.wav"), index=99) == 1


def test_wav_fold_falls_back_to_position_for_unfoldered_corpora():
    # chime backgrounds carry no fold; index decides, and it must stay in 1..5
    folds = [wav_fold(Path(f"chime_bg_{i:03d}.wav"), index=i) for i in range(7)]
    assert folds == [1, 2, 3, 4, 5, 1, 2]


def test_scan_wavs_in_folds_splits_disjointly(tmp_path):
    for fold in (1, 2, 3, 4, 5):
        for take in ("A", "B"):
            sf.write(tmp_path / f"{fold}-1000-{take}-0.wav", np.zeros(16, dtype=np.float32), SR)
    train = scan_wavs_in_folds([tmp_path], [1, 2, 3])
    val = scan_wavs_in_folds([tmp_path], [4])
    fa = scan_wavs_in_folds([tmp_path], [5])
    assert len(train) == 6 and len(val) == 2 and len(fa) == 2
    # the point of the split: no file may appear in two roles
    assert not ({p.name for p in train} & {p.name for p in fa})
    assert not ({p.name for p in val} & {p.name for p in fa})


def test_scan_wavs_in_folds_without_folds_returns_everything(tmp_path):
    sf.write(tmp_path / "1-1000-A-0.wav", np.zeros(16, dtype=np.float32), SR)
    sf.write(tmp_path / "5-1000-A-0.wav", np.zeros(16, dtype=np.float32), SR)
    assert len(scan_wavs_in_folds([tmp_path], [])) == 2
