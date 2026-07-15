# Astra Backend

Go streaming runtime — the serving path of Astra.

## Responsibilities

- Accept client WebSocket connections and ingest PCM audio Frames
- Run VAD and wake word inference via ONNX Runtime (models from `../../assets/models/`)
- Detect utterance endpoints (heuristic over VAD output)
- Orchestrate Whisper and LLM API calls and stream responses back

This service never trains models. See [docs/architecture.md](../../docs/architecture.md) for boundaries.

## Layout

```
cmd/        entrypoints (one main package per binary)
internal/   private application code
configs/    runtime configuration
tests/      integration tests (unit tests live next to the code)
```

## Notes

- Module: `github.com/arnav/astra/services/backend`
- Generated protobuf code lives in `internal/pb/` and is committed; regenerate with `make proto` after editing `proto/astra/v1/astra.proto`.

## Development

From the repo root: `make format`, `make lint`, `make test`.

Run the server: `go run ./cmd/astra` (default `-addr :8080`, WebSocket endpoint at `/`).

Proto codegen prerequisite (once):

```sh
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
```

`protoc` itself comes from Homebrew (`brew install protobuf`).
