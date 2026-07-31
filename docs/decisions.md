# Architectural Decisions

Decision log. Newest at the bottom. Each entry: what we chose, why, and what it rules out.

## 1. Monorepo

One repository holds all services, contracts, docs, and assets.

**Why:** the Go runtime and Python ML code share a contract (ONNX artifacts, protobuf schemas) that must change in lockstep. Separate repos would turn every model or protocol change into a cross-repo coordination problem.

**Rules out:** independent versioning of backend and ML code.

## 2. Separate ML and backend services

`services/ml` and `services/backend` are distinct projects with their own toolchains, dependencies, and tests — they share nothing but the contracts in `proto/` and `assets/`.

**Why:** training and serving have opposite needs (GPU-heavy batch iteration vs. lean low-latency runtime). Coupling them contaminates the serving path with training dependencies.

## 3. ONNX as the training→inference interface

Models are developed in PyTorch, exported to ONNX, and consumed by the Go runtime via ONNX Runtime. Exported artifacts live in `assets/models/`.

**Why:** a portable, versionable artifact decouples model development from the serving stack. Either side can change independently as long as the ONNX contract holds.

**Rules out:** TorchScript serving, embedding Python in the runtime, framework-specific serving stacks.

## 4. Go owns the runtime

All serving-path code — ingest, inference, endpointing, API orchestration — is Go.

**Why:** predictable latency (no GIL, cheap concurrency per Stream), single static binary deployment. The serving path never imports Python.

## 5. Python owns model development

All training, evaluation, and export code is Python.

**Why:** the ML ecosystem (PyTorch, OpenWakeWord, dataset tooling) is Python-native. Fighting that buys nothing.

## 6. Streaming-first architecture

Every pipeline stage consumes and produces Frame streams. Buffering a whole Utterance is allowed only where semantically required (the Whisper hand-off).

**Why:** the project's entire value is latency. Batch-shaped internals would make the latency budget unrecoverable later; retrofitting streaming is a rewrite.

## 7. WebSocket transport with protobuf message schemas

Clients stream PCM Frames over WebSocket; message framing is defined by protobuf schemas in `proto/`. Not gRPC.

**Why:** browsers speak WebSocket natively — no grpc-web proxy layer. Protobuf still gives typed, versioned, cross-language messages.

**Rules out:** gRPC bidirectional streaming as the client protocol.

## 8. Endpointing is a heuristic, not a model

Endpoint detection is a state machine over VAD output (trailing-silence duration, minimum speech length), implemented in the Go runtime with thresholds in config.

**Why:** VAD-derived endpointing is the industry norm, tunable without retraining, and adds zero inference cost. A dedicated semantic end-of-turn model is v2 territory if heuristics prove insufficient.

## 9. Ruff only for Python formatting and linting

`ruff format` + `ruff check`; no Black.

**Why:** `ruff format` is a drop-in Black replacement — one tool, one config, faster. Running both is redundant.

## 10. VAD post-processing is a frame-counting hysteresis state machine, with the Python implementation as the reference

Raw per-frame VAD probabilities are smoothed by a two-state machine (IDLE ↔ SPEECH) combining three techniques: hysteresis (enter speech at `onset_threshold` = the exported `recommended_threshold`, leave only below a lower `offset_threshold`), frame counting (`min_speech_frames` consecutive onset frames to start, `min_silence_frames` consecutive silence frames — the hangover — to stop), and retroactive event emission (`speech_start` points at the first frame of the onset run, `speech_end` at the first frame of the silence run). Frames, not wall-clock timers: deterministic under jitter, replayable offline, unit-testable.

The Python implementation (`services/ml/src/astra_ml/postproc.py`) is the reference; the Go runtime port (the Sink behind `services/backend/internal/server/sink.go`, Phase 3) must match it exactly. All four parameters ship in the model sidecar (`assets/models/vad/vad_v1.json`, `postproc` block) — both sides read the same file; defaults live in `export.py` (`DEFAULT_POSTPROC`). When porting, golden-test parity: feed both implementations one fixed probability sequence, assert identical segments.

Metrics are split in two layers, deliberately: `evaluation/eval.py` stays raw frame-level (measures the model; post-processing would mask regressions and make frame scores history-dependent), while `evaluation/segment_eval.py` scores post-processed events (segment recall, false alarms per hour, onset latency — measures what the user experiences). Tuning the four knobs means grid-searching against segment_eval on the dev split (`evaluation/tune_postproc.py`, objective: lowest false-alarms/hour subject to a recall floor; `--update-sidecar` writes the winner back), then re-exporting the sidecar.

**Rules out:** wall-clock timers in endpointing logic, tuning post-processing knobs against frame-level metrics, implementing post-processing logic in Go first (Python reference always leads).

## 11. Stock OpenWakeWord for the wake word, with our own head trainer

The wake word ("Astraa", pronounced ASS-trah, spelled Astra/Astraa interchangeably) uses OpenWakeWord's architecture: the two frozen frontends (melspectrogram + Google speech-embedding, ONNX, opset 13) with only a small classifier head (`models/ww.py`: 16×96 features → MLP) trained on synthetic data. Positives come from piper-sample-generator's ONNX-voice path (`en_US-libritts_r-medium`, 904 speakers × length/noise-scale grid, resampled to 16 kHz) plus the user's real recordings pitch/speed-augmented; negatives from OWW's precomputed ACAV100M features (~16 GB, memory-mapped), adversarial TTS near-misses, and local audio. The head trains in our own plain-PyTorch loop (`training/train_ww.py`) — OWW's `train.py` was rejected because it imports from a cloned repo layout and pulls py3.12-hostile extras. Real recordings are split **by session, never by clip** (frozen test sessions never trained on or tuned against; family voices eval-only).

**Why:** wake word detection from scratch with synthetic-only positives is a known false-accept trap; OWW's shared embedding carries the large-negative-corpus suppression. Learning-by-building value was already banked in Phase 2.

**Rules out:** custom wake-word architectures in v1; OWW's training pipeline as a dependency; per-clip train/eval splits of recordings.

## 12. Wake word is VAD-gated with preroll backfill

The wake detector scores audio only between VAD `speech_start` and `speech_end` (`internal/wakeword.Sink` runs `vad.Detector` synchronously per frame — no channel between them, so event/frame ordering is deterministic). A ~64-frame ring buffer backfills from `speech_start − preroll_frames` (1 s) because the scoring window is zero-initialized at gate open; on `speech_end` the partial 80 ms chunk is dropped, the window resets, and the trigger machine's run counter clears while its refractory clock (absolute 20 ms frames) survives. Trigger semantics (threshold/patience/refractory) follow the decision #10 recipe: Python reference `postproc_ww.py` leads, Go port golden-tested, knobs in the sidecar, `tune_ww` grid-searches them. Gating semantics themselves are pinned by a Python reference simulator (`GatingSim` in `export/golden_ww.py`) via `ww_gating_golden.json`.

**Why:** gating is the point of the pipeline — downstream stages idle until speech; it also cuts false accepts on non-speech noise, and eval measures the gated configuration so numbers match production.

**Rules out:** always-on wake scoring; wall-clock refractory timers; scoring state that leaks across speech segments.

## 13. Wake word ships as one merged ONNX graph

`export_ww.py` stitches melspec → mel-transform/unfold glue → embedding (batch 16) → head into a single `ww_v1.onnx` (`audio [1, 31840] → prob [1,1]`, opset 13 throughout — the head is exported at the frontends' opset so no version conversion happens). A parity gate (200 random windows vs the Python reference, tol 1e-4; measured 1.8e-7) guards the merge; any failure auto-falls back to a chain-of-3 layout behind the same `ww_v1.json` sidecar (`"graph"` field), which the Go runtime currently rejects explicitly since the fallback never fired. Each score is a pure function of the last `window_samples` of audio — no cross-step feature state, which kills the mel-buffering off-by-one failure class. Two contract traps pinned in the sidecar: the model consumes **raw int16 sample values as float32** (not [-1,1] like VAD), and the mel transform `x/10+2` lives in the merged graph's glue (outside the stock melspec model).

**Why:** one artifact keeps the VAD-established contract (one .onnx + one .json sidecar), one ORT session per process, and the simplest Go integration.

**Rules out:** chain-of-3 serving while merge parity holds; streaming mel/embedding buffer state in the runtime.

## 14. Endpointing is a grace-timer over VAD events, not a second silence machine

`internal/endpoint.Machine` (per stream) reuses the VAD `EventEnd` — which already carries the ~680 ms hangover — as the endpoint signal, and adds a short grace timer on top: after `EventEnd`, `grace_frames` of continued silence closes the `Utterance`, but a new `EventStart` within the window reopens it, so a natural mid-turn pause doesn't cut the speaker off (total turn-end silence ≈ 34 + 15 frames ≈ 0.98 s). It is not a second hysteresis machine over raw probs. The machine hangs off the sinks' existing `onVad`/`onWake` callbacks plus a new per-frame `OnFrame(seq, pcm)` hook (the buffer/clock source) — the VAD/wake sinks stay at the `Server.NewSink` seam, unchanged in structure. Two modes fixed by the runtime wiring: VAD-only arms on `EventStart`; wake-word mode arms on the wake `Event` (a bare `EventStart` only reopens during grace, never arms from idle). Guards: `min_utterance_frames` drops sub-floor blips silently, `max_utterance_frames` force-closes. Knobs are runtime policy in `assets/configs/endpoint.json`, not the model sidecars. The closed `Utterance` (buffered PCM + span metadata) fires an `onUtterance` Go callback — WAV dumper for verification now, Phase 6 ASR later. All frame-counted, no wall-clock.

**Why:** VAD-derived endpointing is the industry norm and the VAD already computes trailing silence; a grace timer decouples "segment end" from "turn end" for ~3 lines instead of a parallel state machine. Reusing the pre-wired nil callbacks matches the Phase 3–4 seam pattern.

**Rules out (for now):** a second silence state machine over raw probs; client-facing `SpeechStarted`/`SpeechEnded` WebSocket events (nothing consumes them yet — no proto change); a dedicated semantic end-of-turn model (v2). Upgrade path if turn-taking latency bites: reduce VAD `min_silence_frames`, or decouple endpointing silence entirely.

## 15. ASR is a per-stream serial worker over an OpenAI-compatible Whisper API

Phase 6 sends closed utterances to a hosted Whisper endpoint (NagaAI, `POST {base_url}/audio/transcriptions`, OpenAI-compatible multipart) instead of running ASR locally. `internal/asr` owns it: a per-stream `Worker` goroutine drains a buffered channel (cap 8) so transcription is serial and in-order per stream and the frame path never blocks; `Enqueue` is non-blocking and drops the newest on overflow. Audio ships as an in-memory 44-byte-header WAV (`endpoint.WavBytes` — no compression; utterances are 100–300 KB, revisit if Phase 8 measures upload as a real cost). Failure policy: 30 s timeout, one retry after 500 ms, then log-and-drop — ASR failure is not stream corruption, the stream lives. Stream close drains the queue in the background (last words still transcribe; socket teardown never waits). The transcript feeds an `onTranscript` seam (log consumer today, Phase 7 LLM later), mirroring how ASR itself plugged into `onUtterance`. Config in `assets/configs/asr.json` (`base_url`/`model`/`language` — pinned `en`, empty means provider auto-detect); key from `NAGA_API_KEY` via env or repo-root `.env` (godotenv — first non-stdlib runtime dep beyond websocket/protobuf/ORT, taken over a hand-rolled parser). ASR is mandatory at startup unless `-no-asr` is explicit.

**Why:** hosted Whisper keeps Phase 6 at integration size — no local inference runtime, no model management — and the free tier is good enough to wire the pipeline end-to-end before latency tuning. Serial-per-stream preserves turn order for Phase 7 conversation chaining; drop-newest on overflow because a full queue means the conversation is already minutes behind and recovery belongs to Phase 8 latency work, not queue juggling.

**Rules out:** local Whisper inference in v1; audio compression before upload; per-utterance fire-and-forget goroutines (unordered transcripts); silent no-key fallback; retry storms (more than one retry poisons queue latency for a turn the user already gave up on).
