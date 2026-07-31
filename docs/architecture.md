# Architecture

Astra is a streaming speech pipeline. A client streams microphone audio to a Go backend, which runs on-device speech models and orchestrates external APIs. Terms used here are defined in the project [glossary](../CONTEXT.md).

## Pipeline

```
Client microphone
    │  Frames (PCM) over WebSocket, protobuf-framed
    ▼
┌─────────────────────────────────────────────┐
│ Go Streaming Backend (services/backend)     │
│                                             │
│  1. Audio ingest      WebSocket session,    │
│                       Frame validation      │
│  2. VAD               ONNX inference,       │
│                       speech prob per Frame │
│  3. Wake Word         OpenWakeWord model    │
│                       (ONNX), gates the     │
│                       rest of the pipeline  │
│  4. Endpointing       heuristic state       │
│                       machine over VAD      │
│                       output; closes the    │
│                       Utterance             │
└─────────────────────────────────────────────┘
    │  completed Utterance audio
    ▼
Whisper API  ──▶  Transcript
    │
    ▼
LLM API  ──▶  response streamed back to client
```

### Stage notes

- **VAD** emits a speech probability per Frame. It makes no boundary decisions — that separation keeps the model simple and the boundary logic tunable without retraining.
- **Wake Word** detection runs on the Stream and gates everything downstream: no wake word, no transcription, no API spend.
- **Endpointing** is not a model. It is a state machine over the VAD probability sequence (sustained trailing silence, minimum speech duration). Thresholds live in runtime config.
- **Whisper and the LLM are external APIs.** Astra's scope ends at producing a clean, well-bounded Utterance and relaying the response; it is a speech front-end, not an LLM host.

## Service boundaries

| | `services/backend` (Go) | `services/ml` (Python) |
|---|---|---|
| Owns | Runtime: ingest, inference, endpointing, API orchestration | Model development: training, evaluation, export |
| Runs | Always (serving path) | Offline only (development path) |
| Never | Trains or defines models | Serves traffic |

**ONNX is the only interface between the two.** Python exports models to `assets/models/`; Go loads them via ONNX Runtime. No shared code, no Python in the serving path.

## Contracts

- `proto/` — protobuf message schemas for the client↔backend WebSocket protocol (audio Frames up, events and responses down).
- `assets/models/` — versioned ONNX artifacts, the ML→backend handoff point.
- `assets/configs/` — runtime configuration (endpointing thresholds, model paths, API settings).

## Latency posture

Streaming-first: every stage consumes Frames as they arrive. Nothing buffers a full Utterance except the final hand-off to Whisper, which requires one. The latency budget is measured wake-word-to-first-response-token.
