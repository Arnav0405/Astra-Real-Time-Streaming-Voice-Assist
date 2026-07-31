---
name: astra-project
description: Astra project — streaming speech front-end monorepo; key scaffold decisions and conventions
metadata: 
  node_type: memory
  type: project
  originSessionId: 91cd2173-8e7c-4cc9-84b6-13fa9e74c004
---

Astra: real-time streaming speech front-end (mic → Go backend → VAD → wake word → endpointing → Whisper API → LLM API). Repo root is `/Users/datafuel/Arnav/voice_assistant` (dir name ≠ project name). Scaffolded 2026-07-14, initial commit a21434f.

Key decisions (full log in docs/decisions.md, but note: user gitignored `docs/` and `CONTEXT.md` — they exist on disk only):
- Go module: `github.com/arnav/astra/services/backend`
- Python 3.12 pinned in services/ml (ML wheel compat), uv-managed, Ruff-only (no Black)
- Client transport: WebSocket + protobuf message schemas in proto/ (not gRPC)
- Endpointing: heuristic over VAD output in Go runtime, no third model
- ONNX artifacts in assets/models/ are the only ML↔backend interface

Next: Phase 1 = WebSocket audio ingest (roadmap in docs/development-roadmap.md). See [[user-commits-themselves]].
