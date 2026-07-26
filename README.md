# Astra

Real-time streaming speech front-end for conversational AI.

Astra handles everything between a live microphone and an LLM: streaming audio ingest, voice activity detection, custom wake-word spotting, utterance endpointing, and (next) transcription. The focus is low-latency speech processing — the LLM itself is a pluggable API at the end of the pipeline.

---

## Where it stands today

I set out to build the hard, unglamorous part of a voice assistant — the front-end that decides *when someone is talking, whether they meant to, and when they're done* — and to do it as a real streaming system, not a demo that buffers a whole clip and calls a cloud API. Five phases in, the pipeline runs end to end, from a real microphone to a captured utterance, entirely in-process:

```
Microphone (client)
    │  20 ms PCM frames, protobuf-framed, over WebSocket
    ▼
Go Streaming Backend  ── all in-process, ONNX Runtime, no per-frame allocs
    │
    ├─▶ ✅ Audio ingest      strict Frame protocol (16 kHz mono s16le, seq-checked)
    ├─▶ ✅ Voice Activity     custom PyTorch model → ONNX, mel frontend baked in
    ├─▶ ✅ Wake Word ("Astraa") OpenWakeWord head, VAD-gated with preroll backfill
    ├─▶ ✅ Endpointing        state machine: arm → capture → grace → closed utterance
    ├─▶ ✅ Whisper API        per-stream serial worker → hosted transcription → transcript seam
    │
    ▼
🔜 LLM API ──▶ 🔜 Client Response
```

And it isn't hand-waved — every model boundary is **golden-tested for Python↔Go parity**, down to per-frame probabilities within `1e-4`.

### What's live

- **Streaming WebSocket ingest** — `StreamStart` → validated 20 ms Frames → `StreamStop`. Any protocol violation returns a typed `Error` and closes the connection. Frames drain into a per-session `Sink` — the seam everything downstream plugs into.
- **Custom VAD, trained from scratch** — 40-mel → 2×Conv1d → GRU(64) → sigmoid, one frame per step, exported to ONNX (opset 17) with the mel frontend baked into the graph (PCM in, prob out — no STFT op at runtime). Trained on LibriParty + CHiME-Home negatives with SNR augmentation. On held-out test: **AUC 0.982 · F1 0.958 · 96.4% segment recall · 11.6 false alarms/hour · p90 onset latency 320 ms**. Post-processing is a hysteresis + debounce machine, grid-searched over 480 configs under a ≥0.95-recall constraint.
- **Custom wake word "Astraa"** — TTS + real-recording positives, ACAV/adversarial negatives, an OpenWakeWord head trained over frozen frontends, all merged into a single `ww_v1.onnx`. On the frozen real-recording test sessions it clears every gate: **recall 0.96 quiet / 0.90 noisy, ~0.9 estimated production false-accepts/hour, median latency 0 ms.** In Go it runs VAD-gated with a 1 s ring-buffer preroll backfill so the trigger never misses the onset.
- **Utterance endpointing** — a per-stream state machine that arms on the wake word (or on speech onset in VAD-only mode), captures PCM, and closes the turn on VAD silence plus a short grace window. A mid-sentence pause shorter than grace doesn't cut the speaker off; a cough or false fire shorter than `min_utterance_frames` is dropped silently. All timing is frame-counted (20 ms/frame), no wall-clock.
- **Whisper transcription** — closed utterances post to a hosted Whisper API (NagaAI, OpenAI-compatible) as in-memory WAVs. A per-stream worker transcribes serially and in order without ever blocking the frame path; failures retry once then drop (the stream lives); stream close drains the queue so the last words still transcribe. The transcript feeds an `onTranscript` seam — logged today, Phase 7's LLM plugs in there. `-no-asr` runs the front-end offline.
- **Manual test loop** — a real mic client (`clients/mic`) streams your voice to the server; a `-verbose` trace prints the VAD/wake/utterance boundaries as they fire, transcripts land in the server log, and `-endpoint-wav-dir` dumps one `.wav` per captured utterance for playback. See [clients/mic/README.md](clients/mic/README.md).

### The discipline behind it

- **ONNX is the only thing that crosses the Python↔Go line.** Python trains and exports; Go loads and runs. Neither reaches into the other.
- **Parity is proven, not assumed.** Every export ships golden fixtures (`astra_ml.export.golden*`), and the Go side asserts against them: exact post-processing event parity, per-frame probs within `1e-4`, and full-stack end-to-end WebSocket tests on real audio chunks.
- **Failure policy is explicit.** Single inference errors are logged and the frame skipped; 50 consecutive failures kill the stream cleanly. VAD is mandatory — the server refuses to start without its runtime and model.

---

## What's next

The front-end is done and speech becomes text; now it earns its keep by talking to a brain.

- **🔜 Phase 7 — LLM + response streaming.** Wire the transcript to an LLM API and stream the response back to the client.
- **🔜 Phase 8 — End-to-end latency.** Measure and tune time-to-first-token across the whole chain — the number this project is ultimately judged by.
- **🔜 Phase 9 — Packaging.** Docker, deployment, the boring-on-purpose infrastructure.

Full phase detail lives in [docs/development-roadmap.md](docs/development-roadmap.md).

---

## Architecture

Two services, one contract:

- **`services/backend`** (Go) owns the runtime: WebSocket audio ingest, ONNX model inference, endpointing, and downstream API orchestration.
- **`services/ml`** (Python) owns model development: training, evaluation, and export of the VAD and wake-word models.
- **`clients/mic`** (Python) is the manual mic test harness.
- **ONNX is the boundary.** Exported artifacts live in `assets/models/`.

See [docs/architecture.md](docs/architecture.md) for detail, [docs/decisions.md](docs/decisions.md) for the decision log, and [CONTEXT.md](CONTEXT.md) for the glossary.

## Repository layout

```
services/backend/   Go streaming runtime (cmd/, internal/{server,vad,wakeword,endpoint}, tests/)
services/ml/        Python model development (training, evaluation, export, models, tests)
clients/mic/        Manual microphone test client
assets/models/      Exported ONNX artifacts (vad/, wakeword/) + sidecar configs
assets/configs/     Shared runtime configuration (endpoint.json)
proto/              Protobuf schemas for the WebSocket audio protocol
docs/               Architecture, roadmap, and decision records
scripts/            Development and CI helper scripts
docker/             Container definitions
```

## Technology stack

| Concern | Choice |
|---|---|
| Streaming runtime | Go |
| Model development | Python 3.12, PyTorch |
| Train/inference boundary | ONNX (Runtime via `yalue/onnxruntime_go`) |
| Wake word | OpenWakeWord (custom "Astraa" model) |
| Transcription | Whisper API *(next)* |
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

Running the pipeline against a live mic (needs the ONNX Runtime shared lib — `brew install onnxruntime`):

```sh
cd services/backend
go run ./cmd/astra -verbose \
  -ww-model ../../assets/models/wakeword/ww_v1.onnx \
  -endpoint-wav-dir /tmp/utt
# then, in another terminal:
services/ml/.venv/bin/python clients/mic/mic_client.py
```

## Development philosophy

- **Streaming-first.** Every pipeline stage consumes and produces streams; nothing buffers a whole utterance unless the stage semantically requires it.
- **Hard service boundary.** Go never trains; Python never serves. ONNX artifacts are the only thing that crosses.
- **Latency is the feature.** Design decisions are judged by their effect on time-to-first-token.
- **Boring infrastructure.** Standard tools, explicit contracts, documented decisions.

## License

[MIT](LICENSE)
