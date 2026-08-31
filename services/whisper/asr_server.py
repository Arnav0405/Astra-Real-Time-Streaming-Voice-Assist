"""gRPC server for streaming ASR transcription."""

from __future__ import annotations

import logging

import grpc

from astra.v1 import asr_pb2, asr_pb2_grpc
from stream_transcriber import StreamingTranscriber

logger = logging.getLogger(__name__)


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

        async for req in request_iterator:
            if req.HasField("config"):
                if config is not None:
                    logger.warning("Received duplicate config, ignoring")
                    continue
                config = req.config
                transcriber = StreamingTranscriber(config.model, config.language)
                transcriber.start()
                logger.info("Started streaming transcription for utterance %d", config.utterance_id)

            elif req.HasField("chunk"):
                if transcriber is None:
                    logger.error("Received chunk before config")
                    continue

                try:
                    text, is_final = transcriber.push(req.chunk.pcm)
                    if text:
                        yield asr_pb2.TranscribeResponse(text=text, is_final=is_final)
                        logger.debug("Partial transcript: %s (final=%s)", text, is_final)
                except Exception as e:
                    logger.exception("Error during transcription push: %s", e)
                    yield asr_pb2.TranscribeResponse(text="", is_final=True)
                    return

        # Finalize and send final transcript
        if transcriber is not None:
            try:
                text, _ = transcriber.finalize()
                yield asr_pb2.TranscribeResponse(text=text, is_final=True)
                logger.info("Finalized transcription for utterance %d: %s", config.utterance_id, text)
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