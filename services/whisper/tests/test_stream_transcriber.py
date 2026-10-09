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


def test_finalize_does_not_resend_the_already_reported_overlap():
    """The final transcript is appended to the partials by the Go client, so it
    must contain only audio that was never transcribed. Re-decoding the 1 s
    overlap made the final a paraphrase of the partial — measured:
    partial "Let's take 1 2 3 4 5 6" then final 'four by six.'."""
    t = StreamingTranscriber("small", "en")
    t.start()
    sizes: list[int] = []
    t._transcribe_chunk = lambda pcm: sizes.append(len(pcm)) or "x"  # type: ignore[method-assign]

    frame = b"\x00\x00" * (640 // 2)
    for _ in range(150):  # one full chunk
        t.push(frame)
    assert sizes == [150 * 640]

    for _ in range(20):  # fresh audio after the chunk boundary
        t.push(frame)
    final, is_final = t.finalize()
    assert is_final is True and final == "x"
    # 20 fresh frames + the 1 s (50 frame) overlap hanging in the buffer,
    # minus the 1 s that was already reported.
    assert sizes == [150 * 640, 20 * 640]


def test_finalize_transcribes_everything_when_no_chunk_was_reported():
    """A short utterance never reaches chunk_bytes, so nothing was reported and
    the whole buffer is new."""
    t = StreamingTranscriber("small", "en")
    t.start()
    sizes: list[int] = []
    t._transcribe_chunk = lambda pcm: sizes.append(len(pcm)) or "x"  # type: ignore[method-assign]

    for _ in range(100):  # 2 s, below the 3 s chunk size
        t.push(b"\x00\x00" * (640 // 2))
    t.finalize()
    assert sizes == [100 * 640]


def test_start_resets_the_reported_overlap_flag():
    """start() runs once per utterance; _chunk_reported must not leak into the
    next one, or a short second utterance would lose its first second."""
    t = StreamingTranscriber("small", "en")
    t.start()
    sizes: list[int] = []
    t._transcribe_chunk = lambda pcm: sizes.append(len(pcm)) or ""  # type: ignore[method-assign]

    for _ in range(150):  # one chunk, so _chunk_reported becomes True
        t.push(b"\x00\x00" * (640 // 2))
    t.finalize()

    t.start()
    for _ in range(100):  # a second, short utterance
        t.push(b"\x00\x00" * (640 // 2))
    t.finalize()
    assert sizes[-1] == 100 * 640