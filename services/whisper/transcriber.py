"""faster-whisper integration for Astra's local ASR service."""

from __future__ import annotations

import io
import os
import threading

import ctranslate2
from faster_whisper import WhisperModel

MODEL_SIZE = "small"
LANGUAGE = "en"

# WHISPER_DEVICE overrides auto-detection ("cuda" | "cpu") — useful for
# testing the fallback path on GPU machines.
_device_override = os.environ.get("WHISPER_DEVICE", "").strip().lower()


def _pick_device_and_compute() -> tuple[str, str]:
    if _device_override in {"cuda", "cpu"}:
        device = _device_override
    elif ctranslate2.get_cuda_device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


_model: WhisperModel | None = None
_lock = threading.Lock()


def get_model() -> WhisperModel:
    """Load the model once, lazily, on first use."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                device, compute_type = _pick_device_and_compute()
                kwargs: dict = {
                    "device": device,
                    "compute_type": compute_type,
                }
                if device == "cpu":
                    kwargs["cpu_threads"] = 4
                _model = WhisperModel(MODEL_SIZE, **kwargs)
    return _model


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe WAV bytes using faster-whisper, forced to English."""
    model = get_model()
    segments, _info = model.transcribe(io.BytesIO(audio_bytes), language=LANGUAGE)
    return " ".join(segment.text.strip() for segment in segments).strip()
