"""gRPC server for streaming ASR transcription."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import grpc

from astra.v1 import asr_pb2, asr_pb2_grpc
from stream_transcriber import StreamingTranscriber

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "assets",
    "configs",
    "asr.json",
)
CONFIG_PATH_ENV = "ASTRA_ASR_CONFIG_PATH"

# Thread pool for Whisper inference (CPU-bound, releases GIL)
_transcriber_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="whisper")


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


class ASRServicer(asr_pb2_grpc.ASRServicer):
    """gRPC servicer for streaming transcription."""

    async def StreamTranscribe(
        self, request_iterator, context: grpc.aio.ServicerContext
    ):
        """Bidirectional streaming transcription.

        Client sends StreamConfig first, then AudioChunk messages.
        Server responds with TranscribeResponse messages (partial + final).
        """
        config = None
        transcriber = None
        cfg = _load_config(
            os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH)
        )

        async for req in request_iterator:
            if req.HasField("config"):
                if config is not None:
                    logger.warning("Received duplicate config, ignoring")
                    continue
                config = req.config
                transcriber = StreamingTranscriber(
                    config.model,
                    config.language,
                    chunk_frames=cfg["chunk_frames"],
                    overlap_frames=cfg["overlap_frames"],
                )
                transcriber.start()
                logger.info("Started streaming transcription for utterance %d", config.utterance_id)

            elif req.HasField("chunk"):
                if transcriber is None:
                    logger.error("Received chunk before config")
                    continue
                logger.info("Received chunk for utterance %d, size=%d bytes", config.utterance_id, len(req.chunk.pcm))

                try:
                    # Run transcriber.push in thread pool to avoid blocking gRPC event loop
                    text, is_final = await asyncio.get_event_loop().run_in_executor(
                        _transcriber_executor, transcriber.push, req.chunk.pcm
                    )
                    if text:
                        logger.info("Partial transcript for utterance %d: %s (final=%s)", config.utterance_id, text, is_final)
                        yield asr_pb2.TranscribeResponse(text=text, is_final=is_final)
                        logger.debug("Partial transcript: %s (final=%s)", text, is_final)
                except Exception as e:
                    logger.exception("Error during transcription push: %s", e)
                    yield asr_pb2.TranscribeResponse(text="", is_final=True)
                    return

# Finalize and send final transcript
        if transcriber is not None:
            try:
                text, _ = await asyncio.get_event_loop().run_in_executor(
                    _transcriber_executor, transcriber.finalize
                )
                logger.info("Finalized transcription for utterance %d: %s", config.utterance_id, text)
                yield asr_pb2.TranscribeResponse(text=text, is_final=True)
            except Exception as e:
                logger.exception("Error during finalize: %s", e)
                yield asr_pb2.TranscribeResponse(text="", is_final=True)


async def serve_grpc(port: int = 50051) -> None:
    """Start the gRPC server."""
    server = grpc.aio.server()
    asr_pb2_grpc.add_ASRServicer_to_server(ASRServicer(), server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("Starting gRPC server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()