# syntax=docker/dockerfile:1

# Astra local Whisper service: faster-whisper behind a FastAPI endpoint.
# Build from repo root:
#   docker build -f services/whisper/Dockerfile -t astra-whisper .

FROM python:3.12-slim-bookworm

# ffmpeg: faster-whisper shells out to it for decoding non-raw audio inputs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# uv: fast, lockfile-driven installs (mirrors the host workflow).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer: cached unless the lockfile changes.
COPY services/whisper/pyproject.toml services/whisper/uv.lock ./
RUN uv sync --frozen --no-install-project --compile-bytecode

# App layer.
COPY services/whisper/main.py services/whisper/transcriber.py \
     services/whisper/asr_server.py services/whisper/stream_transcriber.py \
     services/whisper/silence.py ./

# Generated gRPC stubs: asr_server imports `astra.v1`, so the package must be in
# the image too (the host resolves it from the service's own directory).
COPY services/whisper/astra ./astra/

# 8000 serves /health; 50051 is the gRPC streaming ASR the Go backend dials.
EXPOSE 8000 50051

CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
