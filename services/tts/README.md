# Astra TTS — local Piper voice service

Python gRPC service wrapping [Piper](https://github.com/rhasspy/piper) for fully-local speech synthesis. One `Synthesize` RPC per sentence, PCM streamed back. Design: `docs/superpowers/specs/2026-09-06-local-piper-tts-design.md`.

## API

`astra.v1.TTS/Synthesize` (server-streaming), defined in `proto/astra/v1/tts.proto`:

- Request: `SynthesizeRequest{text, utterance_id}` — one sentence; empty/blank text → `INVALID_ARGUMENT`.
- Stream: one `AudioStart{sample_rate_hz}` first, then zero or more `TtsAudioChunk{pcm, seq}` (s16le mono, whole samples, `seq` 0-based), then the server closes.
- Cancellation: a client that cancels the RPC (barge-in) abandons the stream; the server discards the remaining chunks.

## Config

`assets/configs/tts.json` is shared with the Go backend: `grpc_address` + `sample_rate_hz` are enforced client-side; `voice` and prosody belong to this server.

| Env | Default | Meaning |
|-----|---------|---------|
| `ASTRA_TTS_CONFIG_PATH` | `<repo>/assets/configs/tts.json` | config file location |
| `ASTRA_TTS_MODEL_DIR` | `<repo>/assets/models/tts` | voice model directory |
| `TTS_PORT` | `50052` | listen port |

Docker note: inside the compose network the backend must dial the service name (`tts:50052`), not the shipped `localhost:50052` — same convention as `asr.json` for whisper.

## Run

```sh
uv sync
make tts-voice            # fetch en_US-lessac-medium (~65MB, gitignored)
uv run python server.py
```

## Test

```sh
uv run python -m pytest tests/ -v
```

Synthesizer tests need the model file (skipped otherwise); server tests use a fake.

## Regenerate stubs

After editing `proto/astra/v1/tts.proto`:

```sh
uv run --project services/whisper python -m grpc_tools.protoc -I proto --python_out=services/tts --grpc_python_out=services/tts proto/astra/v1/tts.proto
```

(run from the repo root; `grpcio-tools` is already a dependency of `services/whisper`)
