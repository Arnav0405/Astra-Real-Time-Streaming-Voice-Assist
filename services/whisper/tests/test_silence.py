"""Pure-function tests for the silence gate. No model, no audio files."""

import numpy as np

from silence import (
    FRAME_BYTES,
    SILENCE_RMS,
    SILENCE_RUN_FRAMES,
    frame_rmses,
    is_silent,
    trim_trailing_silence,
)


def speech(frames: int, amplitude: int = 3000) -> bytes:
    """A square wave loud enough to count as speech, `frames` frames long.

    FRAME_BYTES is 640 bytes = 320 int16 samples, so each half-wave is 160
    samples; building 320-sample halves would silently double every count.
    """
    half = FRAME_BYTES // 4
    sample = np.concatenate([np.full(half, amplitude), np.full(half, -amplitude)])
    return sample.astype(np.int16).tobytes() * frames


def silence(frames: int) -> bytes:
    return b"\x00\x00" * (FRAME_BYTES // 2) * frames


def test_frame_rmses_one_value_per_whole_frame():
    rms = frame_rmses(speech(4) + b"\x00")
    assert len(rms) == 4


def test_silence_is_below_the_floor():
    assert is_silent(silence(150)) is True
    assert float(frame_rmses(silence(1))[0]) < SILENCE_RMS


def test_speech_is_above_the_floor():
    assert is_silent(speech(150)) is False


def test_trim_removes_a_long_trailing_silence_run():
    pcm = speech(10) + silence(35)  # 700 ms of tail, what the VAD hangover gives
    trimmed = trim_trailing_silence(pcm)
    assert len(trimmed) == 10 * FRAME_BYTES


def test_trim_leaves_a_short_silence_run_alone():
    pcm = speech(10) + silence(SILENCE_RUN_FRAMES - 1)
    assert trim_trailing_silence(pcm) == pcm


def test_trim_does_not_touch_interior_silence():
    pcm = speech(5) + silence(20) + speech(5)
    assert trim_trailing_silence(pcm) == pcm


def test_trim_is_a_no_op_on_empty_input():
    assert trim_trailing_silence(b"") == b""
