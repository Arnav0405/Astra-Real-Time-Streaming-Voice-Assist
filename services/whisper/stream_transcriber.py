"""Streaming transcription using faster-whisper for real-time ASR."""

from __future__ import annotations

import os
import threading

import numpy as np
import ctranslate2
from faster_whisper import WhisperModel

# WHISPER_DEVICE overrides auto-detection ("cuda" | "cpu")
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
_model_lock = threading.Lock()


def _get_model(model_size: str) -> WhisperModel:
    """Load the model once, lazily, on first use."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                device, compute_type = _pick_device_and_compute()
                kwargs: dict = {
                    "device": device,
                    "compute_type": compute_type,
                }
                if device == "cpu":
                    kwargs["cpu_threads"] = 4
                _model = WhisperModel(model_size, **kwargs)
    return _model


class StreamingTranscriber:
    """Streaming transcription with chunked inference and overlap."""

    def __init__(self, model_size: str = "small", language: str = "en"):
        self.model_size = model_size
        self.language = language
        self.chunk_frames = 150  # 3 seconds at 20ms/frame
        self.overlap_frames = 50  # 1 second overlap
        self.frame_bytes = 640  # 320 samples * 2 bytes (16-bit)
        self.chunk_bytes = self.chunk_frames * self.frame_bytes  # 96000 bytes
        self.overlap_bytes = self.overlap_frames * self.frame_bytes  # 32000 bytes

        self._buffer = bytearray()
        self._started = False

    def start(self) -> None:
        """Initialize the transcriber for a new utterance."""
        self._buffer.clear()
        self._started = True

    def push(self, pcm: bytes) -> tuple[str, bool]:
        """Push PCM frames and return partial transcript.

        Args:
            pcm: Raw s16le PCM bytes (multiple of 640 bytes)

        Returns:
            Tuple of (transcript_text, is_final). is_final is False for partials.
        """
        if not self._started:
            raise RuntimeError("Transcriber not started. Call start() first.")

        self._buffer.extend(pcm)

        # Process chunks when we have enough data
        if len(self._buffer) >= self.chunk_bytes:
            # Take the last chunk_size bytes for inference
            chunk_start = len(self._buffer) - self.chunk_bytes
            chunk = bytes(self._buffer[chunk_start:chunk_start + self.chunk_bytes])

            # Run inference on this chunk
            text = self._transcribe_chunk(chunk)

            # Keep overlap for next chunk
            self._buffer = self._buffer[-self.overlap_bytes:]

            return text, False

        return "", False

    def finalize(self) -> tuple[str, bool]:
        """Finalize transcription and return final transcript."""
        if not self._started:
            raise RuntimeError("Transcriber not started. Call start() first.")

        if len(self._buffer) == 0:
            self._started = False
            return "", True

        # Transcribe remaining buffer
        text = self._transcribe_chunk(bytes(self._buffer))
        self._buffer.clear()
        self._started = False
        return text, True

    def _transcribe_chunk(self, pcm: bytes) -> str:
        """Transcribe a chunk of PCM audio using faster-whisper.

        Accepts raw s16le PCM bytes and converts them directly to a float32
        ndarray, which faster-whisper's transcribe() accepts as input
        without an intermediate WAV/decode step.
        """
        model = _get_model(self.model_size)

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _info = model.transcribe(
            audio,
            language=self.language,
            condition_on_previous_text=False,  # Each chunk independent
            word_timestamps=False,
        )

        return " ".join(segment.text.strip() for segment in segments).strip()


