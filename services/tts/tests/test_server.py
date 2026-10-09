"""Contract tests for server.TTSServicer against a fake synthesizer."""

import grpc
import pytest

from astra.v1 import tts_pb2
from server import TTSServicer


class FakeAbort(Exception):
    pass


class FakeContext:
    """Minimal gRPC aio context: cancelled flag + abort capture."""

    def __init__(self):
        self._cancelled = False
        self.aborted = None

    def cancelled(self):
        return self._cancelled

    async def abort(self, code, details=""):
        self.aborted = (code, details)
        raise FakeAbort()


class FakeSynth:
    """Deterministic ramp: 32768 bytes at 22050 Hz (4 chunks of 8192)."""

    def __init__(self, pcm=None, rate=22050):
        self.pcm = pcm if pcm is not None else bytes(range(256)) * 128
        self.rate = rate

    def synthesize(self, text):
        assert text.strip(), "server must never forward blank text"
        return self.pcm, self.rate


async def collect(servicer, text, ctx):
    out = []
    async for msg in servicer.Synthesize(
        tts_pb2.SynthesizeRequest(text=text, utterance_id=7), ctx
    ):
        out.append(msg)
    return out


@pytest.mark.asyncio
async def test_audio_start_first_then_chunks_in_order():
    msgs = await collect(TTSServicer(FakeSynth()), "hi.", FakeContext())
    assert msgs[0].HasField("audio_start")
    assert msgs[0].audio_start.sample_rate_hz == 22050
    chunks = [m.audio_chunk for m in msgs[1:]]
    assert len(chunks) == 4  # 32768 / 8192
    assert [c.seq for c in chunks] == [0, 1, 2, 3]
    assert b"".join(c.pcm for c in chunks) == FakeSynth().pcm
    assert all(len(c.pcm) % 2 == 0 for c in chunks)


@pytest.mark.asyncio
async def test_blank_text_aborts_invalid_argument():
    ctx = FakeContext()
    with pytest.raises(FakeAbort):
        await collect(TTSServicer(FakeSynth()), "   ", ctx)
    assert ctx.aborted[0] == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_client_cancel_abandons_stream():
    ctx = FakeContext()
    ctx._cancelled = True  # client already gone before first yield
    assert await collect(TTSServicer(FakeSynth()), "hi.", ctx) == []
