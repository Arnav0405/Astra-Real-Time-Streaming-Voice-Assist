"""gRPC server for local Piper TTS: Synthesize(text) -> AudioStart, chunk*."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from astra.v1 import tts_pb2, tts_pb2_grpc
from synthesizer import Synthesizer

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "configs", "tts.json"
)
CONFIG_PATH_ENV = "ASTRA_TTS_CONFIG_PATH"
MODEL_DIR_ENV = "ASTRA_TTS_MODEL_DIR"

# Thread pool for Piper inference (CPU-bound, releases GIL).
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="piper")

# Whole-sample-aligned server chunking; the Go client re-chunks anyway.
CHUNK_BYTES = 8192


def _model_dir() -> str:
    if MODEL_DIR_ENV in os.environ:
        return os.environ[MODEL_DIR_ENV]
    return os.path.join(os.path.dirname(__file__), "..", "..", "assets", "models", "tts")


class TTSServicer(tts_pb2_grpc.TTSServicer):
    """gRPC servicer: one Synthesize RPC synthesizes one sentence."""

    def __init__(self, synth: Synthesizer):
        self._synth = synth

    async def Synthesize(self, request, context):
        text = request.text.strip()
        if not text:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "empty text")
        logger.info("Synthesize utterance=%d chars=%d", request.utterance_id, len(text))
        pcm, rate = await asyncio.get_running_loop().run_in_executor(
            _executor, self._synth.synthesize, text
        )
        if context.cancelled():
            return  # barge-in: client gone, discard
        yield tts_pb2.SynthesizeResponse(
            audio_start=tts_pb2.AudioStart(sample_rate_hz=rate)
        )
        for seq, i in enumerate(range(0, len(pcm), CHUNK_BYTES)):
            if context.cancelled():
                return  # barge-in mid-stream: stop immediately
            yield tts_pb2.SynthesizeResponse(
                audio_chunk=tts_pb2.TtsAudioChunk(pcm=pcm[i : i + CHUNK_BYTES], seq=seq)
            )


async def serve(port: int = 50052) -> None:
    """Load config + voice, serve forever. Fatal on missing model (loud, like VAD)."""
    with open(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH)) as f:
        cfg = json.load(f)
    model_path = os.path.join(_model_dir(), cfg["voice"] + ".onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"voice model missing: {model_path} (run make tts-voice)")
    synth = Synthesizer(
        model_path,
        length_scale=cfg.get("length_scale", 1.0),
        noise_scale=cfg.get("noise_scale", 0.667),
        noise_w=cfg.get("noise_w", 0.333),
    )
    server = grpc.aio.server()
    tts_pb2_grpc.add_TTSServicer_to_server(TTSServicer(synth), server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("TTS serving on :%d voice=%s", port, cfg["voice"])
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(port=int(os.environ.get("TTS_PORT", "50052"))))
