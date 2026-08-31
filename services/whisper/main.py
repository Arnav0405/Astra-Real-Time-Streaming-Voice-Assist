"""Astra local Whisper service: gRPC streaming ASR + HTTP health."""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

from asr_server import serve_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local Whisper Service")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


_grpc_task: asyncio.Task | None = None


@app.on_event("startup")
async def startup() -> None:
    global _grpc_task
    _grpc_task = asyncio.create_task(serve_grpc(50051))
    logger.info("gRPC server task started")


@app.on_event("shutdown")
async def shutdown() -> None:
    global _grpc_task
    if _grpc_task is not None:
        _grpc_task.cancel()
        try:
            await _grpc_task
        except asyncio.CancelledError:
            pass
    logger.info("gRPC server stopped")