"""The bug this pins: every utterance reaches ASR with ~700 ms of trailing
silence (the VAD's hangover), and Whisper decodes silence as boilerplate — "Thank
you.", "You", "Thanks for watching!" — which the Go client appends to the
user's transcript and then answers out loud.

Uses the cached whisper-small model. Skips when the cache is cold so CI without
the model does not fail.
"""

import re
import wave
from pathlib import Path

import pytest

from stream_transcriber import StreamingTranscriber

RECORDING = Path(__file__).resolve().parent.parent / "recordings" / "test.wav"
FRAME = 640
CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-small"

BOILERPLATE = re.compile(r"\b(thank you|thanks for watching|you|please subscribe)\b", re.I)


def frames(pcm: bytes):
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        yield pcm[i : i + FRAME]


def transcribe_utterance(pcm: bytes) -> tuple[list[str], str]:
    t = StreamingTranscriber("small", "en")
    t.start()
    partials = [text for text in (t.push(f)[0] for f in frames(pcm)) if text]
    final, _ = t.finalize()
    return partials, final


@pytest.mark.skipif(not CACHE.exists(), reason="whisper-small not in the HF cache")
def test_speech_with_trailing_silence_has_no_boilerplate_tail():
    with wave.open(str(RECORDING), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        speech = w.readframes(w.getnframes())

    tail = b"\x00\x00" * (320 * 35)  # 700 ms, the VAD hangover + grace the Go side sends
    partials, final = transcribe_utterance(speech + tail)

    whole = " ".join(partials + [final]).strip()
    assert whole, "the utterance contains real speech and must transcribe"
    assert not BOILERPLATE.search(whole), f"ASR invented a tail: {whole!r}"
    # The plan guessed "testing ... six"; the recording actually decodes to
    # "Let's take 1 2 3 4 5 6".
    assert "let's take" in whole.lower() and "1 2 3 4 5 6" in whole.lower(), whole


@pytest.mark.skipif(not CACHE.exists(), reason="whisper-small not in the HF cache")
def test_silence_only_utterance_produces_no_transcript():
    partials, final = transcribe_utterance(b"\x00\x00" * 320 * 200)  # 4 s of digital silence
    assert partials == [] and final == ""
