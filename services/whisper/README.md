# Astra Local Whisper Service

FastAPI wrapper around [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
that replaces the hosted Naga.ai Whisper API. Runs on the same host as the Go
backend, cutting ASR round-trip (TTFS) from ~5s to ~0.3s.

## API

### `GET /health` (HTTP, port 8000)

Liveness probe: `{"status": "ok"}`.

### `astra.v1.ASR/StreamTranscribe` (gRPC, port 50051)

The real surface. The Go backend opens a bidirectional streaming RPC, sends a
`StreamConfig` followed by raw 16 kHz mono s16le `AudioChunk` frames, and
receives partial and final `TranscribeResponse` messages. There is no HTTP
transcription endpoint.

Chunking for the streaming decoder is configured by
`assets/configs/asr.json` (`chunk_frames`, `overlap_frames`), read by this
service at startup. The Go backend loads the same repo-root file through its
`-asr-config` flag.

## Running

```sh
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The model (`whisper-small`) loads lazily on first request; first call also
downloads it (~460MB) into the HuggingFace cache.

## Device selection

Auto-detected at startup: CUDA + float16 when a GPU is present, CPU + int8
(4 threads) otherwise.

Set `WHISPER_DEVICE=cpu|cuda` to force one — e.g. to exercise the fallback
path on a GPU machine.

| Path | Warm latency (test.wav) |
|------|------------------------|
| CUDA / float16 | ~0.27s |
| CPU / int8 | ~1.8s |

## Docker

The image is CPU-only; inside the container auto-detection falls back to the
int8 path since no GPU is exposed.

```sh
docker build -f services/whisper/Dockerfile -t astra-whisper .   # from repo root
docker compose up whisper                                        # or everything
```

## Files

- `main.py` — FastAPI app, endpoints
- `transcriber.py` — model loading and transcription
- `pyproject.toml` / `uv.lock` — dependencies (managed with uv)

The Go backend points here via `assets/configs/asr.json`
(`base_url: http://localhost:8000`). In Docker Compose, use the service name
instead: `http://whisper:8000`.
