"""Astra local Whisper service: POST /audio/transcriptions with WAV bytes."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from transcriber import transcribe_audio

app = FastAPI(title="Local Whisper Service")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/audio/transcriptions")
async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    """Accept WAV bytes and return transcription text."""
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="empty audio upload")
    try:
        text = await run_in_threadpool_safe(audio_data)
    except Exception as exc:  # noqa: BLE001 - surface inference failures as 500
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    return {"text": text}


async def run_in_threadpool_safe(audio_data: bytes) -> str:
    import anyio

    return await anyio.to_thread.run_sync(transcribe_audio, audio_data)
