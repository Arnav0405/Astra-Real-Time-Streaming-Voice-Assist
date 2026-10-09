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
    assert text == ""
    final_text, is_final = transcriber.finalize()
    assert is_final is True
    assert isinstance(final_text, str)


def test_silence_chunk_transcribes_to_empty_string():
    """Whisper decodes silence as boilerplate ("Thank you.", "You"). The gate
    must keep silence away from the decoder entirely."""
    t = StreamingTranscriber("small", "en")
    t.start()
    text, is_final = t.push(b"\x00\x00" * 48000)  # exactly one 3 s chunk
    assert text == ""
    assert is_final is False
    final, is_final = t.finalize()
    assert final == ""
    assert is_final is True