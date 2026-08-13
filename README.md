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
         ▲
         └── ✅ every stage above is timed and streamed back as a waterfall
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
- **Latency, measured** — every stage boundary is stamped server-side and streamed back as a `Turn` message, which the browser draws as a waterfall the moment the reply starts speaking, plus one JSON line per chain for offline aggregation (`scripts/latency.py`). The instrumentation is a tap, not a rewrite: `internal/metrics` is driven entirely from wrappers in `cmd/astra` and no stage package gained a parameter, a field, or an import. Retroactive event indices (the VAD reports the frame speech *began* on, not the one it worked that out on) resolve through a ring of frame arrival times, so the detectors' own lag is charged honestly instead of vanishing. Numbers below.
- **Manual test loop** — a real mic client (`clients/mic`) streams your voice to the server; a `-verbose` trace prints the VAD/wake/utterance boundaries as they fire, transcripts land in the server log, and `-endpoint-wav-dir` dumps one `.wav` per captured utterance for playback. See [clients/mic/README.md](clients/mic/README.md).

### Latency, measured end to end

Every boundary below is stamped **server-side**, so both chains end at a socket write. What the browser does after that — decode, schedule, play — runs on a different clock, and synchronising two clocks to quote a number the server cannot observe would be worse than saying this plainly: **client playback latency is not included.**

Method: the recorded utterance replayed through the real pipeline at realtime pacing (20 ms frames, as a microphone delivers them), VAD-only mode, against the live NagaAI free tier — `whisper-large-v3:free`, `llama-3.3-70b-instruct:free`, `eleven-multilingual-v2:free`. 14 turns. Because it is one clip replayed, `user_speech` and `endpoint_tail` are near-constant by construction; the three network spans are not.

| span | what it covers | p50 | p90 |
| --- | --- | ---: | ---: |
| `vad_detect` | speech onset → the VAD says so | 59 ms | 60 ms |
| `user_speech` | the person talking — *measured, not latency* | 8880 ms | 8880 ms |
| `endpoint_tail` | last speech frame → utterance closed (VAD hangover + grace) | 939 ms | 940 ms |
| `asr` | utterance closed → transcript on the wire | 1767 ms | 2519 ms |
| `llm_ttft` | transcript → first reply token | 624 ms | 1077 ms |
| `tts_ttfb` | first token → first audio byte written | 1808 ms | 2043 ms |
| **time to first audio** | **`endpoint_tail + asr + llm_ttft + tts_ttfb`** | **5046 ms** | **6527 ms** |

The headline deliberately excludes `user_speech` (how long the tester talked is not latency) and `vad_detect` (reported separately as the "does it hear me" number), and deliberately *includes* `endpoint_tail`, because that second is ours: it is the VAD's 680 ms hangover plus a 300 ms grace window, both frame-counted policy in `assets/configs/endpoint.json`.

**Barge-in — the sharp one.** 10 interruptions, same replay method. This chain is pure local computation (no provider is involved between the user's onset and the `Cancel`), so it was measured against a local stub reply, which changes nothing about it:

| span | what it covers | p50 | p90 |
| --- | --- | ---: | ---: |
| `barge_detect` | speech onset → barge-in confirmed | 159 ms | 160 ms |
| `cancel_send` | confirmed → `Cancel` written to the socket | 0 ms | 0 ms |
| **onset to cancel** | | **159 ms** | **160 ms** |

That 159 ms is almost entirely deliberate: the VAD needs 4 frames of speech to declare onset and the endpoint machine holds for `barge_in_frames` (6) before believing it, so a cough or a burst of residual echo cannot cut the assistant off. The server's own work in that chain rounds to zero. It is measured from the frame the user *actually started speaking on*, recovered through the arrival ring — stamping at confirmation time instead would have reported this as ~0 ms and meant nothing.

**The verdict: `tts_ttfb` dominates**, which is what makes local Piper the next piece of work rather than a guess — see [What's next](#whats-next). `asr` is second, and its escape hatch is free when wanted: pointing `asr.json`'s `base_url` at a local `faster-whisper-server` needs zero Go changes.

Reproduce: run the server with stderr redirected to a file, talk to it, then `python3 scripts/latency.py runs/latency.jsonl`.

### The discipline behind it

- **ONNX is the only thing that crosses the Python↔Go line.** Python trains and exports; Go loads and runs. Neither reaches into the other.
- **Parity is proven, not assumed.** Every export ships golden fixtures (`astra_ml.export.golden*`), and the Go side asserts against them: exact post-processing event parity, per-frame probs within `1e-4`, and full-stack end-to-end WebSocket tests on real audio chunks.
- **Failure policy is explicit.** Single inference errors are logged and the frame skipped; 50 consecutive failures kill the stream cleanly. VAD is mandatory — the server refuses to start without its runtime and model.

---

## What's next

The loop closes, and it is now measured. What is left is acting on the measurement.

- **🔜 Phase 9 — Local Piper TTS.** The waterfall says TTS time-to-first-byte is the top span, so that is the one to fix: Piper runs as ONNX, on the same Python↔Go boundary every other model in this repo already crosses, with no network hop and no free-tier ceiling. The cost is espeak-ng phonemization — which is exactly the work the measurement was there to justify.
- **🔜 Phase 10 — Packaging.** Docker, deployment, the boring-on-purpose infrastructure.

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
