# Astra

Real-time streaming speech front-end for conversational AI.

Astra handles everything between a live microphone and an LLM: streaming audio ingest, voice activity detection, custom wake-word spotting, utterance endpointing, transcription, and a spoken reply you can interrupt. The focus is low-latency speech processing — the LLM itself is a pluggable API in the middle of the loop.

---

## Where it stands today

I set out to build the hard, unglamorous part of a voice assistant — the front-end that decides *when someone is talking, whether they meant to, and when they're done* — and to do it as a real streaming system, not a demo that buffers a whole clip and calls a cloud API. Seven phases in, the loop runs end to end — from a real microphone to a spoken answer you can interrupt mid-sentence:

```
Microphone (browser or Python client)
    │  20 ms PCM frames, protobuf-framed, over WebSocket
    ▼
Go Streaming Backend  ── all in-process, ONNX Runtime, no per-frame allocs
    │
    ├─▶ ✅ Audio ingest      strict Frame protocol (16 kHz mono s16le, seq-checked)
    ├─▶ ✅ Voice Activity     custom PyTorch model → ONNX, mel frontend baked in
    ├─▶ ✅ Wake Word ("Astraa") OpenWakeWord head, VAD-gated with preroll backfill
    ├─▶ ✅ Endpointing        state machine: arm → capture → grace → closed utterance
    ├─▶ ✅ Whisper API        per-stream serial worker → hosted transcription → transcript seam
    ├─▶ ✅ Streaming LLM      SSE reply, sentence-chunked as it generates
    ├─▶ ✅ Streaming TTS      synthesized per sentence, paced to realtime
    │                         ▲
    └─▶ ✅ Barge-in ──────────┘  VAD keeps scoring during playback; sustained
                                 speech cancels the turn mid-sentence
    │
    ▼
Client: plays the reply, flushes it the instant you interrupt
```

And it isn't hand-waved — every model boundary is **golden-tested for Python↔Go parity**, down to per-frame probabilities within `1e-4`.

### What's live

- **Streaming WebSocket ingest** — `StreamStart` → validated 20 ms Frames → `StreamStop`. Any protocol violation returns a typed `Error` and closes the connection. Frames drain into a per-session `Sink` — the seam everything downstream plugs into.
- **Custom VAD, trained from scratch** — 40-mel → 2×Conv1d → GRU(64) → sigmoid, one frame per step, exported to ONNX (opset 17) with the mel frontend baked into the graph (PCM in, prob out — no STFT op at runtime). Trained on LibriParty + CHiME-Home negatives with SNR augmentation. On held-out test: **AUC 0.982 · F1 0.958 · 96.4% segment recall · 11.6 false alarms/hour · p90 onset latency 320 ms**. Post-processing is a hysteresis + debounce machine, grid-searched over 480 configs under a ≥0.95-recall constraint.
- **Custom wake word "Astraa"** — TTS + real-recording positives, ACAV/adversarial negatives, an OpenWakeWord head trained over frozen frontends, all merged into a single `ww_v1.onnx`. On the frozen real-recording test sessions it clears every gate: **recall 0.96 quiet / 0.90 noisy, ~0.9 estimated production false-accepts/hour, median latency 0 ms.** In Go it runs VAD-gated with a 1 s ring-buffer preroll backfill so the trigger never misses the onset.
- **Utterance endpointing** — a per-stream state machine that arms on the wake word (or on speech onset in VAD-only mode), captures PCM, and closes the turn on VAD silence plus a short grace window. A mid-sentence pause shorter than grace doesn't cut the speaker off; a cough or false fire shorter than `min_utterance_frames` is dropped silently. All timing is frame-counted (20 ms/frame), no wall-clock.
- **Whisper transcription** — closed utterances post to a hosted Whisper API (NagaAI, OpenAI-compatible) as in-memory WAVs. A per-stream worker transcribes serially and in order without ever blocking the frame path; failures retry once then drop (the stream lives); stream close drains the queue so the last words still transcribe. The transcript feeds an `onTranscript` seam, which the reply runner plugs into. `-no-asr` runs the front-end offline.
- **Spoken replies, streamed** — a closed transcript starts a turn: the LLM streams tokens, a chunker splits them into sentences at terminal punctuation (requiring a following space, so `3.14` stays intact), and each sentence is synthesized while the *next* is still being generated. Audio is paced to at most 300 ms ahead of realtime — not for its own sake, but because a barge-in `Cancel` queued behind several seconds of buffered audio would feel laggy no matter how fast detection was.
- **Barge-in** — the microphone stays live while the assistant talks, so the VAD keeps scoring every frame. Sustained speech for `barge_in_frames` (120 ms by default) cancels the turn: the in-flight LLM and TTS HTTP requests are aborted at the body read, a `Cancel` tells the client to flush what it has buffered, and the utterance re-arms seeded from a ring buffer so the interrupting words keep their onset. A cough or a burst of residual echo, being shorter than the threshold, is ignored.
- **Browser client** — `clients/web` is a dependency-free page (no npm, no build step) served at `/app/`. It exists for one reason: barge-in needs the mic live while the speaker plays, and `getUserMedia({echoCancellation:true})` is WebRTC's AEC3 for free rather than a DSP module to write and tune. Capture runs in a 16 kHz `AudioContext`, so the browser resamples natively and the worklet emits server-ready frames; playback schedules `AudioBufferSourceNode`s in sequence, which makes the flush exactly "stop every scheduled source".
- **Manual test loop** — a real mic client (`clients/mic`) streams your voice to the server; a `-verbose` trace prints the VAD/wake/utterance boundaries as they fire, transcripts land in the server log, and `-endpoint-wav-dir` dumps one `.wav` per captured utterance for playback. See [clients/mic/README.md](clients/mic/README.md).

### The discipline behind it

- **ONNX is the only thing that crosses the Python↔Go line.** Python trains and exports; Go loads and runs. Neither reaches into the other.
- **Parity is proven, not assumed.** Every export ships golden fixtures (`astra_ml.export.golden*`), and the Go side asserts against them: exact post-processing event parity, per-frame probs within `1e-4`, and full-stack end-to-end WebSocket tests on real audio chunks.
- **Failure policy is explicit.** Single inference errors are logged and the frame skipped; 50 consecutive failures kill the stream cleanly. VAD is mandatory — the server refuses to start without its runtime and model.

---

## What's next

The loop closes: you speak, it answers, and you can cut it off. What is missing is the evidence.

- **🔜 Phase 8 — End-to-end latency.** Instrument every span of a turn (`wake → utterance close → ASR → first LLM token → first TTS audio`) plus the barge-in number (`speech onset → cancel → silence`), render them live as a waterfall in the browser client, and publish the p50/p90 table next to the VAD and wake-word tables above. This is the number the project is ultimately judged by, and it is measured rather than assumed — the decision to keep hosted TTS instead of local Piper is deliberately waiting on it.
- **🔜 Phase 9 — Packaging.** Docker, deployment, the boring-on-purpose infrastructure.

Full phase detail lives in [docs/development-roadmap.md](docs/development-roadmap.md).

---

## Architecture

Two services, one contract:

- **`services/backend`** (Go) owns the runtime: WebSocket audio ingest, ONNX model inference, endpointing, and downstream API orchestration.
- **`services/ml`** (Python) owns model development: training, evaluation, and export of the VAD and wake-word models.
- **`clients/web`** is the browser demo client: microphone, playback, and the echo cancellation barge-in depends on.
- **`clients/mic`** (Python) is the manual mic test harness.
- **ONNX is the boundary.** Exported artifacts live in `assets/models/`.

See [docs/architecture.md](docs/architecture.md) for detail, [docs/decisions.md](docs/decisions.md) for the decision log, and [CONTEXT.md](CONTEXT.md) for the glossary.

## Repository layout

```
services/backend/   Go streaming runtime (cmd/, internal/{server,vad,wakeword,endpoint,asr,llm,tts,turn}, tests/)
services/ml/        Python model development (training, evaluation, export, models, tests)
clients/web/        Browser demo client — mic, playback, barge-in (no build step)
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
| Transcription | Whisper API (OpenAI-compatible) |
| Reply | Streaming chat completions (OpenAI-compatible SSE) |
| Speech synthesis | Streaming TTS (OpenAI-compatible), hosted for now |
| Echo cancellation | The browser's WebRTC AEC3, via `getUserMedia` |
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
go run ./cmd/astra -verbose
```

Then open **<http://localhost:8080/app/>**, click Start, say "Astraa", ask something — and talk over the answer to interrupt it. Use **speakers, not headphones**: the echo canceller is the thing being demonstrated.

Replies need `NAGA_API_KEY` in `.env` (see `assets/configs/{asr,llm,tts}.json` for endpoints and models). Without it:

```sh
go run ./cmd/astra -verbose -no-reply     # transcribe only
go run ./cmd/astra -verbose -no-asr       # front-end only, no network at all
```

The Python mic client still works for scripted testing and keeps the WebSocket root path:

```sh
services/ml/.venv/bin/python clients/mic/mic_client.py
```

## Development philosophy

- **Streaming-first.** Every pipeline stage consumes and produces streams; nothing buffers a whole utterance unless the stage semantically requires it.
- **Hard service boundary.** Go never trains; Python never serves. ONNX artifacts are the only thing that crosses.
- **Latency is the feature.** Design decisions are judged by their effect on time-to-first-token.
- **Boring infrastructure.** Standard tools, explicit contracts, documented decisions.

## License

[MIT](LICENSE)
