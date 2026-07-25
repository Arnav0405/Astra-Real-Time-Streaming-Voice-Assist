# Manual mic test

Stream your mic to the astra server and watch the VAD → wake-word → endpoint
pipeline in the terminal. Two terminals: server + mic client.

## Terminal 1 — server (with pipeline logging)

```bash
cd services/backend

# Wake-word mode (VAD gates OWW; utterance arms on the wake word "Astraa"):
go run ./cmd/astra -verbose \
  -ww-model ../../assets/models/wakeword/ww_v1.onnx \
  -endpoint-wav-dir /tmp/utt

# …or VAD-only mode (utterance arms on speech onset, no wake word needed):
go run ./cmd/astra -verbose -endpoint-wav-dir /tmp/utt
```

Needs the ONNX Runtime shared lib (`brew install onnxruntime`; override with
`-ort-lib` / `ASTRA_ORT_LIB`). `-endpoint-wav-dir` writes one WAV per captured
utterance for playback.

## Terminal 2 — mic client

```bash
services/ml/.venv/bin/python clients/mic/mic_client.py
# options: --url ws://localhost:8080   --device <name|index>
```

Speak, then stop with **`q`** (or Ctrl+Q / Ctrl+C) — it sends StreamStop and
closes cleanly.

## What you'll see (server terminal, wake-word mode)

```
[<id>] VAD  ▶ SPEECH  (frame 41)      # speech detected
[<id>] VAD  ■ silence (frame 78)      # speech ended (~680 ms hangover)
[<id>] VAD  ▶ SPEECH  (frame 120)
[<id>] WAKE 🔔 detected (frame 152) → listening   # wake word → utterance armed
[<id>] VAD  ■ silence (frame 240)
[<id>] UTTR ⏹ listening end — utterance 88 frames (seq 152-240)
[<id>] stream <id>: utterance -> /tmp/utt/<id>_000.wav (88 frames, seq 152-240)
```

- **VAD** logs are debounced *boundaries* (speech-start / silence), not per 20 ms
  frame — per-frame would be 50 lines/sec.
- **Listening starts** at the wake word (wake mode) or at speech onset (VAD-only).
- **Listening ends** when the endpoint machine sees the VAD silence plus its grace
  window (`grace_frames`), then the utterance is handed off (WAV dump here).
- A mid-sentence pause shorter than grace does **not** end the utterance; a cough
  / false fire shorter than `min_utterance_frames` is dropped (no UTTR line, no WAV).

Regenerate the protobuf stub after any `proto/astra/v1/astra.proto` change:

```bash
protoc --proto_path=proto --python_out=clients/mic proto/astra/v1/astra.proto
```
