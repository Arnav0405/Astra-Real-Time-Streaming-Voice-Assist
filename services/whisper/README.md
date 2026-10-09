# Astra Local Whisper Service

gRPC streaming ASR around [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
An early revision of this service was a FastAPI upload endpoint replacing the
hosted Naga.ai REST API; the current surface is the real one: the Go backend
opens one bidirectional streaming RPC per client stream and utterances come back
as partial + final transcripts live. This was one of the changes that took a
basic low-latency Go assistant and inflated it into something much larger.

## API

### `GET /health` (HTTP, port 8000)

Liveness probe: `{"status": "ok"}`.

### `astra.v1.ASR/StreamTranscribe` (gRPC, port 50051)

The real surface. The Go backend opens a bidirectional streaming RPC, sends a
`StreamConfig` followed by raw 16 kHz mono s16le `AudioChunk` frames, and
receives partial and final `TranscribeResponse` messages. There is no HTTP
transcription endpoint.

Chunking for the streaming decoder is configured by
`assets/configs/asr.json` (`chunk_frames` = 150 frames = 3 s, `overlap_frames`
= 50), read by this service at startup. The Go backend loads the same repo-root
file through its `-asr-config` flag.

Partials stream while the utterance is still open, and overlapping chunk text is
deduped word-level on the Go side, so the running transcript grows without
duplication. The last partial routinely lands before utterance close, which is
why the measured `asr` span can round to ~0 ms.

Deliberate behaviors:

- **No retry.** A failed utterance is dropped, not retried — a retried
  transcript arrives after the conversation moved on.
- **Silence is not transcribed.** The buffer's silent tail (the VAD's ~680 ms
  hangover) is trimmed before decode so the decoder never hallucinates
  "Thank you." into it (`silence.py`); a test pins this on real audio.

## Running

```sh
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

`main.py` serves a `/health` HTTP endpoint (port 8000) and starts the gRPC
server (port 50051) as a task. The model (`whisper-small`) loads lazily on the
first request; that first call also downloads it (~460MB) into the HuggingFace
cache.

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

- `main.py` — FastAPI app: `/health` + gRPC server startup
- `asr_server.py` — `ASRServicer.StreamTranscribe` (bidirectional, gRPC aio)
- `stream_transcriber.py` — the `StreamingTranscriber`: chunked partial + final decode
- `transcriber.py` — model loading and device selection
- `silence.py` — silent-tail trimming (anti-hallucination)
- `tests/` — servicer contract tests, streaming-transcriber tests, and the
  real-audio silence hallucination pin
- `pyproject.toml` / `uv.lock` — dependencies (managed with uv)

The Go backend points here via `assets/configs/asr.json`
(`grpc_address: localhost:50051`). In Docker Compose, use the service name
instead: `whisper:50051`.

## Test

This service's pytest suite is not wired into the root `make test` (which runs
only `services/ml`). Run it explicitly — the invocation puts the service dir on
`sys.path`:

```sh
./.venv/Scripts/python.exe -m pytest -q      # Windows
./.venv/bin/python -m pytest -q              # POSIX
```
