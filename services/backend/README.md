# Astra Backend

Go streaming runtime — the serving path of Astra.

## Responsibilities

- Accept client WebSocket connections and ingest PCM audio Frames
- Run VAD and wake word inference via ONNX Runtime (models from `../../assets/models/`)
- Detect utterance endpoints (heuristic over VAD output)
- Orchestrate the turn — Whisper (local gRPC), LLM (hosted SSE), Piper TTS (local gRPC) — and stream responses back
- **Streaming partial transcripts** — Whisper partials are coalesced (word-level suffix/prefix overlap dedup) and streamed to the browser in real time, not just the final. The final transcript seeds the LLM turn; partials render dimmed/italic on the client.
- **Barge-in with preroll** — on a confirmed barge, the endpoint machine's ring-buffered onset is pushed to the ASR gRPC stream as a single chunk (never one byte per message) so the interrupting words are transcribed and seed the new turn. A dead lock bug was found and fixed: the client's send and receive paths no longer share a mutex.
- **Latency instrumentation** — every boundary is stamped server-side and streamed back as a `Turn` message (browser draws the waterfall) plus one JSON line per chain.

This service never trains models. See [docs/superpowers/specs/2026-09-06-local-piper-tts-design.md](../../docs/superpowers/specs/2026-09-06-local-piper-tts-design.md) for the design record and [CONTEXT.md](../../CONTEXT.md) for the glossary.

## Layout

```
cmd/        entrypoints (one main package per binary)
internal/   private application code
configs/    runtime configuration
tests/      integration tests (unit tests live next to the code)
```

## Notes

- Module: `github.com/arnav/astra/services/backend`
- Generated protobuf code lives in `internal/pb/` and is committed; regenerate with `make proto` after editing anything under `proto/astra/v1/` (`astra.proto`, `asr.proto`, `tts.proto`).
- gRPC clients for the local services live in `internal/asr/` and `internal/tts/` (raw `grpc.NewClientStream`, lazy `NewClient` — a client that constructs cleanly proves nothing about reachability).

## Development

From the repo root: `make format`, `make lint`, `make test`.

Run the server: `go run ./cmd/astra` (default `-addr :8080`, WebSocket endpoint at `/`).

Proto codegen prerequisite (once):

```sh
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
```

`protoc` itself comes from Homebrew (`brew install protobuf`).
