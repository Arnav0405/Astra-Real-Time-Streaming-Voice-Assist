"""Dataset-free checks for the wake-word v2 window builder.

Builds a toy corpus with the same on-disk shape as the real one (manifest columns,
fold-encoded ESC-50 filenames, session-split recordings) so the path plumbing is
exercised without any download.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from astra_ml.data.kws_dataset import (
    INT16_SCALE,
    SR,
    KwsWindowBuilder,
    KwsWindows,
    build_pools,
    mix_at_snr,
    place_full,
    place_partial,
    random_crop,
    trim_silence,
)
from astra_ml.training.kws_config import (
    KwsAugmentConfig,
    KwsConfig,
    KwsDataConfig,
    KwsEvalConfig,
    KwsFrontendConfig,
    KwsGatingConfig,
    KwsModelConfig,
    KwsPostprocDefaults,
    KwsTrainingConfig,
)

WINDOW = 15360


def _tone(seconds: float, freq: float = 220.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    tts = tmp_path / "ww_tts"
    rows = []
    for i in range(4):
        _write(tts / "positives" / f"{i}.wav", _tone(0.7, 200 + 20 * i))
        rows.append((f"positives/{i}.wav", "positive", "astraa", "train" if i < 3 else "val"))
    for i in range(3):
        _write(tts / "adversarial" / "astro" / f"{i}.wav", _tone(0.8, 300 + 10 * i))
        rows.append((f"adversarial/astro/{i}.wav", "adversarial", "astro", "train"))
    _write_csv(tts / "manifest.csv", ["path", "label", "phrase", "split"], rows)

    rec = tmp_path / "ww_recordings"
    _write(rec / "s1" / "000.wav", _tone(1.2, 180))
    _write(rec / "s2" / "000.wav", _tone(1.2, 190))
    _write_csv(
        rec / "manifest_split.csv",
        ["path", "session", "speaker", "env", "split"],
        [("s1/000.wav", "s1", "a", "quiet", "train"), ("s2/000.wav", "s2", "a", "quiet", "eval")],
    )

    neg = tmp_path / "ww_negative_recordings"
    _write(neg / "n1" / "000.wav", _tone(1.5, 260))
    _write(neg / "n2" / "000.wav", _tone(1.5, 270))
    _write_csv(
        neg / "manifest_split.csv",
        ["path", "session", "speaker", "env", "split"],
        [("n1/000.wav", "n1", "a", "quiet", "train"), ("n2/000.wav", "n2", "a", "quiet", "eval")],
    )

    # ESC-50 encodes its fold as the leading filename digit; 44.1 kHz on purpose so the
    # resample path in random_crop is exercised.
    for fold in (1, 2, 3, 4, 5):
        _write(tmp_path / "esc50" / f"{fold}-100{fold}-A-0.wav", _tone(5.0, 400, 44100), sr=44100)
    for name in ("train-clean-100", "dev-clean", "test-clean"):
        _write(tmp_path / "librispeech" / name / "1" / "1.wav", _tone(6.0, 150))
    _write(tmp_path / "rirs" / "a.wav", np.array([1.0, 0.3, 0.1], dtype=np.float32))
    return tmp_path


def _cfg(corpus: Path, **training) -> KwsConfig:
    return KwsConfig(
        data=KwsDataConfig(
            tts_out=corpus / "ww_tts",
            recordings_root=corpus / "ww_recordings",
            negative_recordings_root=corpus / "ww_negative_recordings",
            negative_audio_dirs=[corpus / "esc50"],
            negative_folds=[1, 2, 3],
            negative_val_folds=[4],
            speech_negative_dirs=[corpus / "librispeech" / "train-clean-100"],
            speech_negative_val_dirs=[corpus / "librispeech" / "dev-clean"],
        ),
        augment=KwsAugmentConfig(
            rir_dir=corpus / "rirs",
            noise_dirs=[corpus / "esc50"],
            partial_negative_frac=training.pop("partial_negative_frac", 0.0),
        ),
        training=KwsTrainingConfig(runs_dir=corpus / "runs", batch_size=8, **training),
        postproc=KwsPostprocDefaults(threshold=0.85, patience_frames=2, refractory_seconds=2.0),
        gating=KwsGatingConfig(preroll_frames=50, partial_chunk="drop"),
        eval=KwsEvalConfig(
            fa_audio_dirs=[corpus / "esc50"],
            fa_folds=[5],
            recall_floor_quiet=0.95,
            recall_floor_noisy=0.8,
            max_fa_per_hour=20,
            max_latency_ms=500,
        ),
        frontend=KwsFrontendConfig(),
        model=KwsModelConfig(),
    )


def test_trim_silence_finds_the_spoken_span():
    clip = np.concatenate([np.zeros(3200, np.float32), _tone(0.5), np.zeros(3200, np.float32)])
    trimmed = trim_silence(clip)
    assert 0.45 * SR <= trimmed.size <= 0.55 * SR


def test_trim_silence_survives_all_silence():
    silent = np.zeros(1600, dtype=np.float32)
    assert trim_silence(silent).size == silent.size


def test_place_full_keeps_the_whole_keyword_inside():
    rng = np.random.default_rng(0)
    clip_len = 8000
    for _ in range(200):
        p = place_full(clip_len, WINDOW, jitter=3200, rng=rng)
        assert p.n_inside == clip_len
        assert 0 <= p.start <= WINDOW - clip_len


def test_place_full_jitter_spans_enough_scoring_steps():
    """400 ms of jitter must survive as at least 4 distinct 80 ms alignments, or
    postproc's patience_frames=2 has no margin."""
    rng = np.random.default_rng(1)
    starts = {place_full(8000, WINDOW, 3200, rng).start // 1280 for _ in range(500)}
    assert len(starts) >= 4


def test_place_full_crops_when_clip_exceeds_window():
    rng = np.random.default_rng(2)
    p = place_full(WINDOW + 5000, WINDOW, 3200, rng)
    assert p.n_inside == WINDOW
    assert p.start == 0


def test_place_partial_leaves_most_of_the_keyword_out():
    rng = np.random.default_rng(3)
    for _ in range(200):
        p = place_partial(8000, WINDOW, max_frac=0.6, rng=rng)
        assert 0 < p.n_inside <= 0.6 * 8000
        assert p.start >= 0 and p.start + p.n_inside <= WINDOW


def test_mix_at_snr_measures_against_the_keyword_not_the_padding():
    signal = np.zeros(WINDOW, dtype=np.float32)
    region = _tone(0.5)
    signal[:region.size] = region
    bg = np.ones(WINDOW, dtype=np.float32)
    mixed = mix_at_snr(signal, region, bg, snr_db=0.0)
    added = mixed - signal
    # 0 dB against the keyword region, not against a window that is 70% silence.
    assert np.isclose(np.square(added).mean(), np.square(region).mean(), rtol=1e-4)


def test_random_crop_resamples_non_16k_sources(corpus: Path):
    rng = np.random.default_rng(0)
    esc = sorted((corpus / "esc50").glob("*.wav"))[0]
    assert random_crop(esc, rng, WINDOW).shape == (WINDOW,)


def test_random_crop_windows_16k_sources(corpus: Path):
    rng = np.random.default_rng(0)
    long_wav = corpus / "librispeech" / "train-clean-100" / "1" / "1.wav"
    a = random_crop(long_wav, rng, WINDOW)
    b = random_crop(long_wav, rng, WINDOW)
    assert a.shape == (WINDOW,) and b.shape == (WINDOW,)
    assert not np.array_equal(a, b)  # different offsets


def test_builder_emits_a_full_window_from_a_short_clip(corpus: Path):
    cfg = _cfg(corpus)
    builder = KwsWindowBuilder(cfg.augment, WINDOW)
    out = builder.from_clip(_tone(0.6), np.random.default_rng(0))
    assert out.shape == (WINDOW,)
    assert out.dtype == np.float32
    assert np.abs(out).max() <= 1.0


def test_builder_never_emits_digital_silence_padding(corpus: Path):
    """A zero-padded window is a shape the runtime never produces: scoring only happens
    behind an open VAD gate with 1 s of real preroll."""
    cfg = _cfg(corpus)
    builder = KwsWindowBuilder(cfg.augment, WINDOW)
    out = builder.from_clip(_tone(0.4), np.random.default_rng(7))
    assert np.count_nonzero(out[:1000]) > 500


def test_batch_composition_is_positional(corpus: Path):
    cfg = _cfg(corpus)
    ds = KwsWindows(cfg, build_pools(cfg, "train"), length=cfg.training.batch_size)
    labels = [float(ds[i][1]) for i in range(cfg.training.batch_size)]
    assert labels == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 25% of 8 == 2


def test_partial_windows_are_labelled_negative(corpus: Path):
    cfg = _cfg(corpus, partial_negative_frac=1.0)
    ds = KwsWindows(cfg, build_pools(cfg, "train"), length=cfg.training.batch_size)
    assert [float(ds[i][1]) for i in range(2)] == [0.0, 0.0]


def test_items_are_int16_scaled_and_deterministic(corpus: Path):
    cfg = _cfg(corpus)
    ds = KwsWindows(cfg, build_pools(cfg, "train"), length=64)
    window, label = ds[0]
    assert window.shape == (cfg.frontend.window_samples,)
    assert window.dtype == np.float32 and label.dtype == np.float32
    assert 0.1 * INT16_SCALE < np.abs(window).max() <= INT16_SCALE
    np.testing.assert_array_equal(window, ds[0][0])
    assert not np.array_equal(window, ds[cfg.training.batch_size][0])


def test_every_negative_source_is_reachable(corpus: Path):
    cfg = _cfg(corpus)
    ds = KwsWindows(cfg, build_pools(cfg, "train"), length=64)
    assert ds.NEGATIVE_ORDER == ("speech", "adversarial", "environment", "negatives_user")
    for slot in range(cfg.training.batch_size - ds.n_positive):
        window, label = ds[ds.n_positive + slot]
        assert window.shape == (WINDOW,) and float(label) == 0.0


def test_val_pools_are_fold_and_session_disjoint_from_train(corpus: Path):
    cfg = _cfg(corpus)
    train, val = build_pools(cfg, "train"), build_pools(cfg, "val")
    train.check()
    val.check()
    assert not set(train.positives_tts) & set(val.positives_tts)
    assert not set(train.positives_user) & set(val.positives_user)
    assert not set(train.environment) & set(val.environment)
    assert not set(train.speech) & set(val.speech)
