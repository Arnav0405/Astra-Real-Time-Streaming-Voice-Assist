# Astra

Real-time streaming speech front-end for conversational AI.

Astra handles everything between a live microphone and an LLM: streaming audio ingest, voice activity detection, custom wake-word spotting, utterance endpointing, transcription, and a spoken reply you can interrupt. The focus is low-latency speech processing — the LLM itself is a pluggable API in the middle of the loop.

> **Project status: complete.** This started as a side project a basic low-latency assistant in Go, to learn the language by building with it and then it got inflated into microservices and everything. Maybe it deploys to the cloud someday, Kubernetes managing the containers or that happens on an even better project.

---

## The pipeline

The hard part of a voice assistant: deciding *when someone is talking, whether they meant to, and when they're done* — as a real streaming system, not a buffer-then-call-the-cloud demo. Fourteen phases in, the loop runs end to end, mic to spoken answer you can interrupt mid-sentence:

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
    ├─▶ ✅ Local Whisper      per-stream worker → gRPC → transcript seam
    ├─▶ ✅ Streaming LLM      SSE reply, sentence-chunked as it generates
    ├─▶ ✅ Piper TTS          sentence-synthesized at reply time, paced to realtime
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

- **Ingest** — 20 ms PCM frames over WebSocket into a per-session `Sink`; protocol violation → typed `Error`, stream closed.
- **Voice activity** — custom VAD trained from scratch (PyTorch → ONNX, mel frontend baked into the graph): **AUC 0.982 · F1 0.958 · 11.6 false alarms/hour**, hysteresis + debounce post-processing.
- **Wake word "Astraa"** — custom OpenWakeWord head, VAD-gated with 1 s ring-buffer preroll: **recall 0.96 quiet / 0.90 noisy**, median trigger latency 0 ms.
- **Endpointing** — per-stream state machine, frame-counted (no wall-clock); a pause shorter than the grace window never cuts the speaker off.
- **Transcription** — local `faster-whisper` over gRPC (`services/whisper`); partials stream to the browser live, overlap coalesced word-level. `-no-asr` runs offline.
- **Replies, streamed** — hosted LLM (SSE) chunked into sentences; each one synthesized by local Piper TTS over gRPC (`services/tts`) while the next generates; audio paced ≤300 ms ahead of realtime.
- **Barge-in** — VAD keeps scoring during playback; sustained speech (120 ms threshold) cancels the turn mid-sentence, and the ring-buffered preroll goes to the ASR stream as one chunk so the interrupting words are transcribed.

### Latency

Every boundary stamped server-side; both chains end at a socket write; client playback is not included. Method for every column: one recorded utterance replayed through the real pipeline at realtime pacing (20 ms frames), VAD-only mode, aggregated by `scripts/latency.py`. Column 1 is 14 turns (p50; p90 in this README's git history); columns 2–4 are single turns — each number is the one measured value. Hosted columns ran on the NagaAI free tier; the current stack is the OpenCode Go LLM (`mimo-v2.6-flash`) plus local Whisper and Piper on one CPU host.

| span | what it covers | 1 — hosted stack | 2 — local Whisper | 3 — + OpenCode Go LLM | 4 — + local Piper TTS |
| --- | --- | ---: | ---: | ---: | ---: |
| `wake_detect` | speech onset → the VAD says so | ~59 ms | — | — | **0 ms** |
| `user_speech` | the person talking — *measured, not latency* | 8880 ms | 9509 ms | 18998 ms | 18998 ms |
| `endpoint_tail` | last speech frame → utterance closed (VAD hangover + grace) | 939 ms | 678 ms | 679 ms | 679 ms |
| `asr` | utterance closed → transcript on the wire | 1767 ms | 172 ms | — | **2 ms** |
| `llm_ttft` | transcript → first reply token | 624 ms | 2417 ms | — | **2396 ms** |
| `tts_ttfb` | first token → first audio byte written | 1808 ms | 2657 ms | — | **688 ms** |
| **time to first audio** | **`endpoint_tail + asr + llm_ttft + tts_ttfb`** | **5046 ms** | **5924 ms** | — | **3765 ms** |

| # | status | what it shipped | where it landed in the table |
| --- | --- | --- | --- |
| 2 | ✅ | local `faster-whisper` over gRPC replaces the hosted Whisper API | `asr` 1767 → 172 ms |
| 3 | ✅ | LLM provider switch (NagaAI → OpenCode Go, `mimo-v2.6-flash`) | current-method column |
| 4 | ✅ | local Piper TTS over gRPC (`services/tts`), design in [docs/superpowers/specs/2026-09-06-local-piper-tts-design.md](docs/superpowers/specs/2026-09-06-local-piper-tts-design.md) | `tts_ttfb` 2657 → 688 ms |

The remainder — and the top span — is `llm_ttft` (~2.4 s): the hosted LLM is the last network piece of the loop.

Reproduce: run the server with stderr redirected to a file, then `python3 scripts/latency.py runs/latency.jsonl`; from a container, `docker compose logs --no-log-prefix astra > runs/latency.jsonl`.

## Layout

```
services/backend/   Go streaming runtime (cmd/, internal/{server,vad,wakeword,endpoint,asr,llm,tts,turn}, tests/)
services/ml/        Python model development (training, evaluation, export, models, tests)
services/whisper/   Local faster-whisper transcription service (gRPC)
services/tts/       Local Piper TTS service (gRPC, en_US-lessac-medium)
clients/web/        Browser demo client — mic, playback, barge-in (no build step)
clients/mic/        Manual microphone test client
assets/models/      Exported ONNX artifacts (vad/, wakeword/) + sidecar configs
assets/configs/     Shared runtime configuration (endpoint.json, asr.json, llm.json, tts.json)
proto/              Protobuf schemas (WebSocket protocol, ASR gRPC, TTS gRPC)
docs/               Design records (docs/superpowers/)
scripts/            Development and CI helper scripts
docker/             Container definitions
```

ONNX is the only thing that crosses the Python↔Go line. Glossary lives in [CONTEXT.md](CONTEXT.md); each service directory has its own README.

## Run it

```sh
docker compose up
```

Open **<http://localhost:8080/app/>**, click Start, say "Astraa", ask something — and talk over the answer to interrupt it. Use **speakers, not headphones**: the echo canceller is the thing being demonstrated. The image carries the Go binary, ONNX Runtime 1.29.0 (the version the parity fixtures were verified against), the models, and the browser client; amd64 + arm64. Without a key, the offline front-end still runs:

```sh
docker compose run --rm --service-ports astra -no-asr -verbose
```

The microphone stays in the browser — the container only ever sees WebSocket frames, so there is no audio device to pass through; running the container on *another* machine is the only case that needs more (`ssh -L 8080:localhost:8080 user@host`, and the URL stays `localhost`).

From source — needs the ONNX Runtime shared lib (`brew install onnxruntime`) — flags append the same way in both modes:

```sh
cd services/backend
go run ./cmd/astra -verbose
go run ./cmd/astra -verbose -no-reply     # transcribe only
go run ./cmd/astra -verbose -no-asr       # front-end only, no network at all
```

The Python mic client still works for scripted testing (see [clients/mic/README.md](clients/mic/README.md)):

```sh
services/ml/.venv/bin/python clients/mic/mic_client.py
```

Make targets, from the repo root:

```sh
make format && make lint && make test   # Go + Python
make proto                              # after editing anything under proto/astra/v1/
```

## License

[MIT](LICENSE)
