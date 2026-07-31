# Development Roadmap

Implementation phases. Headings only — each phase gets detailed when it starts.

## Phase 0 — Repository scaffold ✅

## Phase 1 — Streaming audio ingest (WebSocket, Frame protocol) ✅

WebSocket endpoint accepting protobuf-framed audio (`proto/astra/v1/astra.proto`): `StreamStart` → validated 20 ms Frames (16 kHz mono s16le, strict seq) → `StreamStop`. Any protocol violation returns an `Error` and closes the connection. Accepted Frames drain into a per-session channel behind the `Sink` interface (`internal/server/sink.go`) — the seam VAD plugs into in Phase 3. Phase 1 sink logs stream stats. Covered by integration tests (`make test`).

## Phase 2 — VAD model: dataset, training, ONNX export ✅

Streaming VAD trained in `services/ml` and exported to `assets/models/vad/vad_v1.onnx` (opset 17) with tuned params in the `vad_v1.json` sidecar. Architecture (`src/astra_ml/models/vad.py`): 40-mel frontend → 2×Conv1d (stride 2) → GRU(64) → Linear → sigmoid, one 20 ms frame per step. The mel frontend is baked into the ONNX graph as fixed DFT/filterbank matmuls (no STFT op), so the runtime contract is pure PCM in, prob out: `pcm [1,320]` + `state_in [1,1,64]` → `prob [1,1]` + `state_out [1,1,64]`.

Trained on LibriParty mixtures with CHiME-Home backgrounds as negatives and runtime SNR noise augmentation (0–20 dB). Frame-level eval (`runs/vad/eval_report.json`): AUC 0.982, F1 0.958. Post-processing is hysteresis + debounce (`src/astra_ml/postproc.py`), grid-searched over 480 combos on dev (`runs/vad/tune_report.json`) under a ≥0.95 recall constraint → onset 0.77 / offset 0.09, min_speech 4 frames, min_silence 34 frames. Segment eval on held-out test (`runs/vad/segment_report.json`): **96.4% segment recall, 11.6 false alarms/hour, p90 onset latency 320 ms** over 450 segments / 0.6 h.

## Phase 3 — VAD inference in the Go runtime ✅

`internal/vad` runs `vad_v1.onnx` in-process via `yalue/onnxruntime_go` (ONNX Runtime shared lib: `brew install onnxruntime`, override with `-ort-lib`/`ASTRA_ORT_LIB`). One shared session per process; per-stream `Inferencer` converts each 640-byte s16le Frame to `float32 [1,320]` in [-1,1] and threads the GRU state `[1,1,64]` between steps. The post-processing hysteresis machine is an exact port of `postproc.py` (decision #10), reading onset/offset/min_speech/min_silence from the `vad_v1.json` sidecar; `vad.Sink` replaces the stats sink at the `Server.NewSink` seam and emits start/end events (retroactive frame indices) through an `onEvent` callback — the seam wake word (Phase 4) and endpointing (Phase 5) consume.

Failure policy: single inference errors are logged and the frame skipped (clock still advances); 50 consecutive failures kill the stream with `internal_error` (the `Sink` interface gained `Fatal()`/`Err()` for this). VAD is mandatory — the server fails at startup if the runtime or model is missing. Parity is golden-tested against Python (`astra_ml/export/golden.py` → `internal/vad/testdata/`): exact postproc event parity, per-frame prob parity within 1e-4, and an end-to-end WebSocket test on a real CHiME speech chunk matching the reference segments exactly (`services/backend/tests/e2e_test.go`).

## Phase 4 — Custom wake word: OpenWakeWord training and integration ✅

Wake word "Astraa" (decisions #11–13). Python side (`services/ml`): TTS positives via piper-sample-generator ONNX voices + user recordings (session-split, frozen test set) + ACAV/adversarial negatives → `train_ww` head training over frozen OWW frontends → `ww_eval`/`tune_ww` (recall quiet/noisy, FA/hour, latency vs gates: ≥95%/≥80% recall, ≤500 ms) → `export_ww` merges frontends+head into single `ww_v1.onnx` + `ww_v1.json` sidecar (parity-gated, chain fallback). Go side: `internal/wakeword` — `Sink` runs `vad.Detector` synchronously, gates the detector on speech with 1 s preroll backfill from a ring buffer, scores 80 ms chunks against a zero-initialized 31840-sample window, trigger machine ported from `postproc_ww.py`; wired at `Server.NewSink` behind `-ww-model`/`-ww-config` (empty model path = VAD-only). Parity: `ww_postproc_golden.json` (exact), `ww_gating_golden.json` (gating rules, exact), `ww_inference_golden.json` (1e-4), full-stack `e2e_ww_golden.json` — fixtures from `astra_ml.export.golden_ww`.

`ww_report.json` on the frozen real-recording test sessions (`quiet_near_0719`, `far_noise_0719`) clears all four gates: recall_quiet 0.96 (≥0.95), recall_noisy 0.90 (≥0.80), fa_per_hour 190.2 (≤200, ACAV-estimated production FA 0.9/h), latency_ms_median 0 ms (≤500). `ww_v1.onnx`/`ww_v1.json` committed under `assets/models/wakeword/`; full-stack e2e parity (`TestEndToEndWakeWord`) passes through the real Go WebSocket pipeline.

## Phase 5 — Endpoint detection (utterance layer) ✅

`internal/endpoint` is the utterance layer: it ties a start trigger to a subsequent VAD `EventEnd`, buffers the PCM of that span, and hands a closed `Utterance` to a server-side consumer. A per-stream `Machine` is fed by the existing sink callbacks — `OnVad`/`OnWake` for boundaries plus a new `OnFrame(seq, pcm)` per-frame hook the VAD/wake-word sinks gained (`internal/vad/sink.go`, `internal/wakeword/sink.go`) — so the sinks stay in place at the `Server.NewSink` seam and endpointing hangs off their callbacks (decision #14). Two modes, fixed per stream by the runtime wiring: VAD-only arms on VAD `EventStart`; wake-word mode arms on the wake `Event` (a bare `EventStart` never arms from idle, so it only reopens during grace).

Close logic reuses the VAD `EventEnd` (which already carries the ~680 ms hangover) plus a short grace timer: after `EventEnd`, `grace_frames` of continued silence closes the utterance, but a new `EventStart` inside the window reopens it — a natural mid-turn pause doesn't cut the speaker off. Total turn-end silence ≈ 34 (VAD hangover) + 15 (grace) frames ≈ 0.98 s. Guards: `min_utterance_frames` drops sub-floor blips (false wake fire, cough) silently; `max_utterance_frames` force-closes a stuck-open utterance. All timing is frame-counted (20 ms/frame), no wall-clock. Knobs live in `assets/configs/endpoint.json` (runtime policy, not model-coupled — not in the VAD/wake sidecars), loaded via `-endpoint-config`. The closed `Utterance` fires an `onUtterance` Go callback wired to a WAV dumper (`-endpoint-wav-dir`, one `.wav` per utterance) for playback verification today; Phase 6 ASR swaps in as the consumer. Client-facing `SpeechStarted`/`SpeechEnded` WebSocket events are deferred (nothing consumes them yet, no proto change). Unit-tested (`internal/endpoint/machine_test.go`): normal close, grace reopen, min-drop, max-timeout, wake-mode arming.

## Phase 6 — Whisper API integration ✅

`internal/asr` turns closed utterances into text via a hosted Whisper endpoint (NagaAI, OpenAI-compatible `POST {base_url}/audio/transcriptions`). Each stream gets a `Worker`: a single goroutine draining a buffered channel (capacity 8), so utterances transcribe serially and in order while the frame-processing path never blocks — `Enqueue` is non-blocking and drops the newest utterance (logged loud) if the queue is somehow full. The `Client` wraps the PCM in an in-memory 44-byte-header WAV (`endpoint.WavBytes`, shared with the dumper) and posts it multipart (`file`, `model`, `language`) with a Bearer key. Failure policy: 30 s per-request timeout, one retry after 500 ms, then the utterance is logged and dropped — ASR failure is not stream corruption, so the stream lives (unlike VAD's 50-strike kill). On stream close the worker drains: queued and in-flight utterances finish transcribing in the background (`drainSink` in `cmd/astra`), so the last words of a session still transcribe without blocking socket teardown.

The transcript lands in a `Transcript` struct (`StreamID`, `Text`, seq span, frame count) handed to an `onTranscript` consumer — today a server-side log line, the seam Phase 7's LLM consumer plugs into, exactly as ASR plugged into `onUtterance`. Config is runtime policy in `assets/configs/asr.json` (`base_url`, `model: whisper-large-v3:free`, `language: en`; timeout/retry/queue-size are code constants until they need tuning), loaded via `-asr-config`. The API key comes from `NAGA_API_KEY` — exported env wins, else loaded from the repo-root `.env` via godotenv (`-env-file`). ASR is mandatory: a missing key is fatal at startup unless `-no-asr` is passed explicitly (offline VAD/wake/endpointing work), never a silent fallback. The WAV dumper coexists (`-endpoint-wav-dir`), chained before the enqueue. Unit-tested against `httptest` (`internal/asr`): multipart/WAV request shape, auth header, retry-then-success, retry exhaustion, timeout, ordering, queue-full drop, drain-on-close, failed-transcription skip. Manual loop: run the server, speak through `clients/mic`, watch transcript logs.

## Phase 7 — LLM API integration and response streaming

## Phase 8 — End-to-end latency measurement and tuning

## Phase 9 — Packaging and deployment (Docker)
