# Astra

Real-time streaming speech front-end for conversational AI.

Astra handles everything between a live microphone and an LLM: streaming audio ingest, voice activity detection, wake word spotting, utterance endpointing, and transcription. The focus is low-latency speech processing — the LLM itself is a pluggable API at the end of the pipeline.

## Architecture

```
Microphone (client)
    │  PCM frames over WebSocket
    ▼
Go Streaming Backend
    │
    ├─▶ Voice Activity Detection   (custom PyTorch model, ONNX runtime)
    ├─▶ Wake Word Detection        (OpenWakeWord, custom wake word)
    ├─▶ Endpoint Detection         (heuristic state machine over VAD output)
    │
    ▼
Whisper API ──▶ LLM API ──▶ Client Response
```

Two services, one contract:

- **`services/backend`** (Go) owns the runtime: WebSocket audio ingest, ONNX model inference, endpointing, and downstream API orchestration.
- **`services/ml`** (Python) owns model development: training, evaluation, and export of the VAD and wake word models.
- **ONNX is the boundary.** Python trains and exports; Go loads and runs. Exported artifacts live in `assets/models/`.

See [docs/architecture.md](docs/architecture.md) for detail and [CONTEXT.md](CONTEXT.md) for the project glossary.

## Repository layout

```
services/backend/   Go streaming runtime (cmd/, internal/, configs/, tests/)
services/ml/        Python model development (training/, evaluation/, export/, models/, tests/)
assets/models/      Exported ONNX artifacts consumed by the backend
assets/configs/     Shared runtime configuration
proto/              Protobuf message schemas for the WebSocket audio protocol
docs/               Architecture, roadmap, and decision records
scripts/            Development and CI helper scripts
docker/             Container definitions
```

## Technology stack

| Concern | Choice |
|---|---|
| Streaming runtime | Go |
| Model development | Python 3.12, PyTorch |
| Train/inference boundary | ONNX |
| Wake word | OpenWakeWord (custom model) |
| Transcription | Whisper API |
| Client transport | WebSocket (protobuf-framed PCM) |
| Python tooling | uv, Ruff, Pytest |

## Development

```sh
make format   # format Go + Python
make lint     # lint Go + Python
make test     # run all test suites
make proto    # generate code from protobuf schemas
make clean    # remove build artifacts
```

## Development philosophy

- **Streaming-first.** Every pipeline stage consumes and produces streams; nothing buffers a whole utterance unless the stage semantically requires it.
- **Hard service boundary.** Go never trains; Python never serves. ONNX artifacts are the only thing that crosses.
- **Latency is the feature.** Design decisions are judged by their effect on time-to-first-token.
- **Boring infrastructure.** Standard tools, explicit contracts, documented decisions ([docs/decisions.md](docs/decisions.md)).

## Roadmap

Audio ingest → VAD → wake word → endpointing → Whisper integration → LLM integration → client response streaming. Phases in [docs/development-roadmap.md](docs/development-roadmap.md).

## License

[MIT](LICENSE)
