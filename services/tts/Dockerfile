# syntax=docker/dockerfile:1

# Astra local Piper TTS: piper-tts behind a gRPC endpoint.
# Build from repo root (after `make tts-voice`):
#   docker build -f services/tts/Dockerfile -t astra-tts .

FROM python:3.13-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv: fast, lockfile-driven installs (mirrors the host workflow).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer: cached unless the lockfile changes.
COPY services/tts/pyproject.toml services/tts/uv.lock ./
RUN uv sync --frozen --no-install-project --compile-bytecode

# App layer: server, synthesizer, generated stubs.
COPY services/tts/server.py services/tts/synthesizer.py ./
COPY services/tts/astra ./astra

# Model layer: baked in. Fails the build loudly if `make tts-voice`
# was not run — a voiceless TTS image must never exist silently.
COPY assets/models/tts/en_US-lessac-medium.onnx* ./models/
ENV ASTRA_TTS_MODEL_DIR=/app/models

EXPOSE 50052

CMD ["uv", "run", "--no-sync", "python", "server.py"]
