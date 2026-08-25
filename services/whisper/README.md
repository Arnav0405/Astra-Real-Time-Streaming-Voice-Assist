# Astra Local Whisper Service

FastAPI wrapper around [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
that replaces the hosted Naga.ai Whisper API. Runs on the same host as the Go
backend, cutting ASR round-trip (TTFS) from ~5s to ~0.3s.

## API

### `POST /audio/transcriptions`

Multipart upload with a `file` field containing WAV bytes:

```sh
curl -X POST http://localhost:8000/audio/transcriptions -F "file=@recording.wav"
# -> {"text": "testing one two three four five six"}
```

Transcription is always forced to English (`language="en"`).

### `GET /health`

Liveness probe: `{"status": "ok"}`.

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
