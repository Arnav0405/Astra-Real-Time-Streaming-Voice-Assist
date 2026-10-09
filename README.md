# Astra

Real-time streaming speech front-end for conversational AI.

Astra handles everything between a live microphone and an LLM: streaming audio ingest, voice activity detection, custom wake-word spotting, utterance endpointing, transcription, and a spoken reply you can interrupt. The focus is low-latency speech processing — the LLM itself is a pluggable API in the middle of the loop.

---

## Where it stands today

The hard part of a voice assistant: deciding *when someone is talking, whether they meant to, and when they're done* — as a real streaming system, not a buffer-then-call-the-cloud demo. Nine phases in, the loop runs end to end, mic to spoken answer you can interrupt mid-sentence, measured at every seam, one command to run:

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

Every model boundary is **golden-tested for Python↔Go parity**, per-frame probabilities within `1e-4`.

### What's live

- **Streaming WebSocket ingest** — `StreamStart` → validated 20 ms Frames → `StreamStop`. Protocol violation → typed `Error`, connection closed. Frames drain into a per-session `Sink`, the seam everything downstream plugs into.
- **Custom VAD, trained from scratch** — 40-mel → 2×Conv1d → GRU(64) → sigmoid, one frame per step, ONNX (opset 17), mel frontend baked into the graph (PCM in, prob out, no STFT at runtime). Trained on LibriParty + CHiME-Home negatives, SNR augmentation. Held-out test: **AUC 0.982 · F1 0.958 · 96.4% segment recall · 11.6 false alarms/hour · p90 onset latency 320 ms**. Post-processing: hysteresis + debounce, grid-searched over 480 configs under ≥0.95-recall constraint.
- **Custom wake word "Astraa"** — TTS + real-recording positives, ACAV/adversarial negatives, OpenWakeWord head over frozen frontends, merged into `ww_v1.onnx`. Frozen real-recording test: **recall 0.96 quiet / 0.90 noisy, ~0.9 est. production false-accepts/hour, median latency 0 ms.** VAD-gated in Go with 1 s ring-buffer preroll backfill so the trigger never misses onset.
- **Utterance endpointing** — per-stream state machine: arms on wake word (or speech onset in VAD-only mode), captures PCM, closes on VAD silence + grace window. Pause shorter than grace doesn't cut the speaker off; false fire shorter than `min_utterance_frames` drops silently. Frame-counted (20 ms/frame), no wall-clock.
- **Whisper transcription** — closed utterances stream over gRPC to the local faster-whisper service (`services/whisper`). Per-stream worker transcribes in order, never blocks the frame path; failures drop the utterance; stream close drains the queue. Feeds `onTranscript` → reply runner. `-no-asr` runs offline.
- **Spoken replies, streamed** — closed transcript starts a turn: LLM streams tokens, chunker splits into sentences at terminal punctuation (requires trailing space, so `3.14` stays intact), each sentence synthesized while the next generates. Audio paced ≤300 ms ahead of realtime, so a barge-in `Cancel` isn't stuck behind seconds of buffered audio.
- **Barge-in** — mic stays live during playback, VAD keeps scoring every frame. Sustained speech for `barge_in_frames` (120 ms default) cancels the turn: in-flight LLM/TTS requests aborted at the body read, `Cancel` tells the client to flush, utterance re-arms seeded from a ring buffer so the interrupting words keep their onset. Cough or echo burst shorter than threshold: ignored. **The preroll is pushed to the ASR stream as one chunk** (a per-byte gRPC message would have exploded the stream), so the words you spoke at the very start of the interruption are transcribed and fed to the new turn.
- **Streaming partial transcripts** — ASR partials stream live to the browser as they arrive; the backend coalesces overlapping Whisper chunks (word-level suffix/prefix match) so the running transcript grows cleanly without duplication. Partials render dimmed/italic; the final replaces them normally.
- **Browser client** — `clients/web`, dependency-free, no build step, served at `/app/`. Exists because barge-in needs the mic live while the speaker plays, and `getUserMedia({echoCancellation:true})` gives WebRTC's AEC3 for free. Capture runs in a 16 kHz `AudioContext` (native resample, server-ready frames from the worklet); playback schedules `AudioBufferSourceNode`s in sequence, so flush = stop every scheduled source.
- **Latency, measured** — every stage boundary stamped server-side, streamed back as a `Turn` message (browser draws it as a waterfall), plus one JSON line per chain for offline aggregation (`scripts/latency.py`). Instrumentation is a tap: `internal/metrics` runs entirely from wrappers in `cmd/astra`, no stage package touched. Retroactive event indices (VAD reports the frame speech *began* on, not the one it worked that out on) resolve through a ring of frame arrival times, so detector lag gets charged honestly. Numbers below.
- **One command to run it** — `docker compose up`, open the page, talk. One image: binary, ONNX Runtime (pinned to the parity-verified version), models, browser client; builds natively on amd64/arm64. Zero Go changes — container paths live in `ENTRYPOINT`, `go run ./cmd/astra` from source unchanged. API key passed in, never baked; missing key, server says so and exits.
- **Manual test loop** — `clients/mic` streams your voice to the server; `-verbose` prints VAD/wake/utterance boundaries as they fire, transcripts land in the server log, `-endpoint-wav-dir` dumps one `.wav` per captured utterance. See [clients/mic/README.md](clients/mic/README.md).

### Latency, measured end to end

Every boundary stamped **server-side**; both chains end at a socket write. **Client playback latency is not included** — decode/schedule/play runs on a different clock the server can't observe.

Method: recorded utterance replayed through the real pipeline at realtime pacing (20 ms frames), VAD-only mode, against the live NagaAI free tier — `whisper-large-v3:free`, `llama-3.3-70b-instruct:free`, `eleven-multilingual-v2:free`. 14 turns. One clip replayed → `user_speech`/`endpoint_tail` near-constant by construction; the three network spans are not.

| span | what it covers | p50 | p90 |
| --- | --- | ---: | ---: |
| `vad_detect` | speech onset → the VAD says so | 59 ms | 60 ms |
| `user_speech` | the person talking — *measured, not latency* | 8880 ms | 8880 ms |
| `endpoint_tail` | last speech frame → utterance closed (VAD hangover + grace) | 939 ms | 940 ms |
| `asr` | utterance closed → transcript on the wire | 1767 ms | 2519 ms |
| `llm_ttft` | transcript → first reply token | 624 ms | 1077 ms |
| `tts_ttfb` | first token → first audio byte written | 1808 ms | 2043 ms |
| **time to first audio** | **`endpoint_tail + asr + llm_ttft + tts_ttfb`** | **5046 ms** | **6527 ms** |

**Revised metrics — local Whisper + pipelined pipeline.** Same 14-turn replay method, now with a local `faster-whisper` server (small model) replacing the hosted Whisper API, and `chunk_frames: 150` (3 s of audio per partial, 1.6 s overlap). LLM and TTS still hosted:

| span | what it covers | p50 | p90 |
| --- | --- | ---: | ---: |
| `vad_detect` | speech onset → the VAD says so | 59 ms | 60 ms |
| `user_speech` | the person talking — *measured, not latency* | 9509 ms | 9509 ms |
| `endpoint_tail` | last speech frame → utterance closed (VAD hangover + grace) | 678 ms | 678 ms |
| `asr` | utterance closed → final transcript on the wire | 172 ms | — |
| `llm_ttft` | transcript → first reply token | 2417 ms | — |
| `tts_ttfb` | first token → first audio byte written | 2657 ms | — |
| **time to first audio** | **`endpoint_tail + asr + llm_ttft + tts_ttfb`** | **5924 ms** | — |

Local Whisper brings `asr` from ~1.7 s down to ~172 ms. The headline latency is now dominated by the hosted LLM and TTS providers — the next improvement target.

Headline excludes `user_speech` (not latency) and `vad_detect` (reported separately), *includes* `endpoint_tail` — that second is ours: VAD's 680 ms hangover + 120 ms grace, frame-counted policy in `assets/configs/endpoint.json`. (Table measured at 300 ms grace; hangover dominates either way.)

**Barge-in — the sharp one.** 10 interruptions, same replay method. Pure local computation (no provider between user onset and `Cancel`), measured against a local stub reply:

| span | what it covers | p50 | p90 |
| --- | --- | ---: | ---: |
| `barge_detect` | speech onset → barge-in confirmed | 159 ms | 160 ms |
| `cancel_send` | confirmed → `Cancel` written to the socket | 0 ms | 0 ms |
| **onset to cancel** | | **159 ms** | **160 ms** |

159 ms is mostly deliberate: VAD needs 4 frames to declare onset, endpoint machine holds `barge_in_frames` (6) before believing it — so a cough or echo burst can't cut the assistant off. Server's own work in that chain rounds to zero. Measured from the frame the user *actually started speaking on*, recovered through the arrival ring.

**Verdict: `tts_ttfb` dominates** — makes local Piper the next piece of work, not a guess — see [What's next](#whats-next). `asr` is second; escape hatch is free: point `asr.json`'s `base_url` at a local `faster-whisper-server`, zero Go changes.

Reproduce: run the server with stderr redirected to a file, talk to it, then `python3 scripts/latency.py runs/latency.jsonl`. From the container, the spans are on stdout — `docker compose logs --no-log-prefix astra > runs/latency.jsonl` (the prefix Compose adds by default is not JSON, and the script parses lines).

### The discipline behind it

- **ONNX is the only thing that crosses the Python↔Go line.** Python trains and exports; Go loads and runs. Neither reaches into the other.
- **Parity is proven, not assumed.** Every export ships golden fixtures (`astra_ml.export.golden*`), and the Go side asserts against them: exact post-processing event parity, per-frame probs within `1e-4`, and full-stack end-to-end WebSocket tests on real audio chunks.
- **Failure policy is explicit.** Single inference errors are logged and the frame skipped; 50 consecutive failures kill the stream cleanly. VAD is mandatory — the server refuses to start without its runtime and model.

---

## What's next

Loop closes, measured, packaged. One item open, and the measurement is what names it:

- **🔜 Phase 9 — Local Piper TTS.** Waterfall says TTS time-to-first-byte is the top span. Piper runs as ONNX, same Python↔Go boundary every other model here crosses, no network hop, no free-tier ceiling. Cost: espeak-ng phonemization. After that, local LLM inference is the final frontier for fully offline, sub-2 s end-to-end latency.

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
| Transcription | Whisper API (OpenAI-compatible) or local `faster-whisper` over gRPC |
| Reply | Streaming chat completions (OpenAI-compatible SSE) |
| Speech synthesis | Streaming TTS (OpenAI-compatible), hosted for now |
| Echo cancellation | The browser's WebRTC AEC3, via `getUserMedia` |
| Client transport | WebSocket (protobuf-framed PCM) |
| Python tooling | uv, Ruff, Pytest |
| Packaging | Docker (multi-stage, `debian:bookworm-slim`, amd64 + arm64) |

## Development

```sh
make format   # format Go + Python
make lint     # lint Go + Python
make test     # run all test suites
make proto    # generate code from protobuf schemas
make clean    # remove build artifacts
make docker   # build the container image
make up       # docker compose up --build
```

### Run it — Docker

Nothing to install but Docker. Put `OPENCODE_GO_KEY` in a repo-root `.env` (Compose reads it automatically; the key never enters the image), then:

```sh
docker compose up
```

Open **<http://localhost:8080/app/>**, click Start, say "Astraa", ask something — and talk over the answer to interrupt it. Use **speakers, not headphones**: the echo canceller is the thing being demonstrated.

The image carries the Go binary, ONNX Runtime 1.29.0 (pinned to the version the parity fixtures were verified against), the exported models and the browser client; it builds natively on both amd64 and arm64. Without a key, the server tells you so and exits — for the offline front-end (VAD, wake word, endpointing, no network at all):

```sh
docker compose run --rm --service-ports astra -no-asr -verbose
```

Any flag appends the same way, because the paths live in the image's `ENTRYPOINT` and the command is empty.

> The microphone stays in the browser — the container only ever sees WebSocket frames, so there is no audio device to pass through. `http://localhost` is a secure context, so `getUserMedia` works over plain HTTP. Running the container on *another* machine is the only case that needs more: reach it as `ssh -L 8080:localhost:8080 user@host` and the URL stays `localhost`, or put it behind a TLS proxy.

### Run it — from source

Running the pipeline against a live mic (needs the ONNX Runtime shared lib — `brew install onnxruntime`):

```sh
cd services/backend
go run ./cmd/astra -verbose
```

Then open **<http://localhost:8080/app/>**, click Start, say "Astraa", ask something — and talk over the answer to interrupt it. Use **speakers, not headphones**: the echo canceller is the thing being demonstrated.

Replies need `OPENCODE_GO_KEY` in `.env` (see `assets/configs/{asr,llm,tts}.json` for endpoints and models). Without it:

```sh
go run ./cmd/astra -verbose -no-reply     # transcribe only
go run ./cmd/astra -verbose -no-asr       # front-end only, no network at all
```

The Python mic client still works for scripted testing and keeps the WebSocket root path:

```sh
services/ml/.venv/bin/python clients/mic/mic_client.py
```

## License

[MIT](LICENSE)
