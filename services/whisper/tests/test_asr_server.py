"""Tests for ASR gRPC server."""

import pytest
import grpc

from astra.v1 import asr_pb2, asr_pb2_grpc
from asr_server import ASRServicer


class MockContext:
    """Mock gRPC context for testing."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.asyncio
async def test_asr_servicer_stream_transcribe():
    """Test the ASR servicer stream transcribe method."""
    servicer = ASRServicer()

    # Create test requests
    config = asr_pb2.TranscribeRequest(
        config=asr_pb2.StreamConfig(
            utterance_id=1,
            sample_rate_hz=16000,
            channels=1,
            bits_per_sample=16,
            model="small",
            language="en",
        )
    )

    # Silence PCM (150 frames = 96000 bytes)
    chunk = asr_pb2.TranscribeRequest(
        chunk=asr_pb2.AudioChunk(pcm=b"\x00\x00" * 48000)  # Half chunk
    )

    # Test the async generator
    async def request_gen():
        yield config
        yield chunk
        yield chunk  # Second chunk to trigger processing

    responses = []
    async for response in servicer.StreamTranscribe(request_gen(), MockContext()):
        responses.append(response)

    # Should get at least one response
    assert len(responses) >= 1
    # Final response should be is_final=True
    assert responses[-1].is_final is True