"""Tests for StreamingTranscriber."""

import pytest
from stream_transcriber import StreamingTranscriber


def test_streaming_transcriber_partials():
    transcriber = StreamingTranscriber("small", "en")
    transcriber.start()
    # Push 150 frames of silence (3000 samples = 6000 bytes for 16-bit)
    pcm = b"\x00\x00" * 3000
    text, is_final = transcriber.push(pcm)
    assert is_final is False
    assert isinstance(text, str)
    final_text, is_final = transcriber.finalize()
    assert is_final is True
    assert isinstance(final_text, str)