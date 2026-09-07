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
COPY services/whisper/main.py services/whisper/transcriber.py services/whisper/asr_server.py ./

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
