"""Silence measurement for the streaming ASR path.

Whisper decodes silence as its training-data boilerplate ("Thank you.", "You"),
and faster-whisper's no-speech guard only drops such a segment when the model is
ALSO unsure of it (no_speech_prob > 0.6 *and* avg_logprob < -1.0) — a confident
hallucination survives. So the service never asks the decoder about silence.
"""

from __future__ import annotations

import numpy as np

# One 20 ms frame of 16 kHz mono s16le.
FRAME_BYTES = 640

# Per-frame RMS in int16 units, below which a frame counts as silence.
# Measured on this project's mic: speech in recordings/test.wav has RMS ~275.
# Re-measure with:
#   ./.venv/Scripts/python.exe -c "import wave,numpy as np; \
#     a=np.frombuffer(wave.open('recordings/test.wav').readframes(10**6),dtype=np.int16); \
#     print(np.sqrt((a.astype(float)**2).mean()))"
SILENCE_RMS = 60.0

# Trailing silence is only trimmed when the run is at least this long (100 ms).
# A shorter run is left alone so a quiet word ending is never clipped.
SILENCE_RUN_FRAMES = 5


def frame_rmses(pcm: bytes) -> np.ndarray:
    """Per-frame RMS (int16 units), one entry per whole frame. Trailing partial
    bytes are ignored."""
    whole = len(pcm) - (len(pcm) % FRAME_BYTES)
    if whole == 0:
        return np.empty(0, dtype=np.float32)
    samples = np.frombuffer(pcm[:whole], dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(samples.reshape(-1, FRAME_BYTES // 2) ** 2, axis=1))


def is_silent(pcm: bytes, floor: float = SILENCE_RMS) -> bool:
    """True when every whole frame in pcm is below the silence floor."""
    rms = frame_rmses(pcm)
    if rms.size == 0:
        return True
    return bool((rms < floor).all())


def trim_trailing_silence(
    pcm: bytes, floor: float = SILENCE_RMS, min_run: int = SILENCE_RUN_FRAMES
) -> bytes:
    """Drop a trailing run of sub-floor frames when it is at least min_run long.

    The utterance handed to ASR ends with the VAD's ~680 ms hangover, and
    decoding that tail is what makes Whisper emit boilerplate.
    """
    rms = frame_rmses(pcm)
    run = 0
    while run < rms.size and rms[rms.size - 1 - run] < floor:
        run += 1
    if run < min_run:
        return pcm
    return pcm[: (rms.size - run) * FRAME_BYTES]
