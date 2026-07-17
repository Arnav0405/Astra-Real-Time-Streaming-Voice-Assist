"""Generate golden fixtures for the Go VAD runtime (Phase 3 parity tests).

Writes two JSON files into services/backend/internal/vad/testdata/:

- postproc_golden.json: synthetic probability sequence + events expected from
  the reference VadPostprocessor (decision #10 — Go must match exactly).
- inference_golden.json: deterministic PCM (base64 s16le) + per-frame probs
  from onnxruntime on the committed vad_v1.onnx (Go must match within 1e-4).

Run from services/ml: .venv/bin/python -m astra_ml.export.golden
"""

import base64
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from astra_ml.postproc import PostprocConfig, VadPostprocessor

REPO_ROOT = Path(__file__).resolve().parents[5]
MODEL_DIR = REPO_ROOT / "assets" / "models" / "vad"
TESTDATA = REPO_ROOT / "services" / "backend" / "internal" / "vad" / "testdata"

FRAME_SAMPLES = 320
N_FRAMES = 100  # 2 s of audio


def postproc_probs(cfg: PostprocConfig) -> list[float]:
    """Sequence exercising every branch of the state machine."""
    hi, lo, mid = cfg.onset_threshold + 0.1, cfg.offset_threshold - 0.05, 0.5
    seq: list[float] = []
    seq += [lo] * 5
    seq += [hi] * (cfg.min_speech_frames - 1)  # onset blip, too short
    seq += [lo] * 3
    seq += [hi] * cfg.min_speech_frames  # real onset -> "start"
    seq += [mid] * 10  # in speech, above offset
    seq += [lo] * (cfg.min_silence_frames - 1)  # silence blip, too short
    seq += [hi] * 2
    seq += [lo] * cfg.min_silence_frames  # real end -> "end"
    seq += [lo] * 5
    seq += [hi] * cfg.min_speech_frames  # second segment...
    seq += [mid] * 5  # ...left open for finish()
    return seq


def gen_postproc(sidecar: dict) -> None:
    cfg = PostprocConfig.from_sidecar(sidecar)
    probs = postproc_probs(cfg)
    pp = VadPostprocessor(cfg)
    events = [[e[0], e[1]] for p in probs if (e := pp.push(p))]
    finish = [[e[0], e[1]] for e in pp.finish()]
    out = {"probs": probs, "events": events, "finish_events": finish}
    (TESTDATA / "postproc_golden.json").write_text(json.dumps(out, indent=1))


def gen_inference(sidecar: dict) -> None:
    rng = np.random.default_rng(0)
    # Deterministic noise bursts separated by silence; prob values are
    # irrelevant for parity, only that Go reproduces them.
    pcm = np.zeros(N_FRAMES * FRAME_SAMPLES, dtype=np.float32)
    pcm[16000:24000] = rng.normal(0, 0.3, 8000)
    pcm[28000:32000] = np.sin(np.arange(4000) * 2 * np.pi * 440 / 16000) * 0.5
    pcm_i16 = np.clip(pcm * 32768, -32768, 32767).astype("<i2")

    sess = ort.InferenceSession(str(MODEL_DIR / "vad_v1.onnx"))
    state = np.zeros(sidecar["state_shape"], dtype=np.float32)
    probs = []
    for i in range(N_FRAMES):
        frame = pcm_i16[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES].astype(np.float32) / 32768.0
        prob, state = sess.run(["prob", "state_out"], {"pcm": frame[None, :], "state_in": state})
        probs.append(float(prob[0, 0]))

    out = {
        "pcm_s16le_base64": base64.b64encode(pcm_i16.tobytes()).decode(),
        "frame_samples": FRAME_SAMPLES,
        "probs": probs,
    }
    (TESTDATA / "inference_golden.json").write_text(json.dumps(out, indent=1))


def gen_e2e(sidecar: dict) -> None:
    """Real speech chunk + expected segment events, for the Go end-to-end test.

    Uses the CHiME eval_speech manifest when the (gitignored) dataset is
    present; the committed fixture keeps the test reproducible elsewhere.
    """
    ml_root = Path(__file__).resolve().parents[3]
    manifest = ml_root / "datasets" / "chime_prepared" / "eval_speech.csv"
    if not manifest.exists():
        print("chime dataset absent; skipping e2e fixture regeneration")
        return
    import soundfile as sf

    sess = ort.InferenceSession(str(MODEL_DIR / "vad_v1.onnx"))
    cfg = PostprocConfig.from_sidecar(sidecar)

    # First manifest chunk that actually yields events — an empty fixture
    # would make the Go e2e test vacuous.
    for line in manifest.read_text().splitlines():
        wav_rel = line.split(",")[1]
        audio, sr = sf.read(Path(__file__).resolve().parents[3] / wav_rel, dtype="int16")
        assert sr == sidecar["sample_rate"], (sr, wav_rel)
        if audio.ndim > 1:
            audio = audio[:, 0]
        n = len(audio) // FRAME_SAMPLES * FRAME_SAMPLES
        audio = audio[:n].astype("<i2")

        state = np.zeros(sidecar["state_shape"], dtype=np.float32)
        pp = VadPostprocessor(cfg)
        events = []
        for i in range(n // FRAME_SAMPLES):
            frame = audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES].astype(np.float32) / 32768.0
            feeds = {"pcm": frame[None, :], "state_in": state}
            prob, state = sess.run(["prob", "state_out"], feeds)
            if e := pp.push(float(prob[0, 0])):
                events.append([e[0], e[1]])
        events += [[e[0], e[1]] for e in pp.finish()]
        if events:
            break
    else:
        raise SystemExit("no manifest chunk produced VAD events")

    out = {
        "source": wav_rel,
        "pcm_s16le_base64": base64.b64encode(audio.tobytes()).decode(),
        "events": events,
    }
    (TESTDATA / "e2e_golden.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    TESTDATA.mkdir(parents=True, exist_ok=True)
    sidecar = json.loads((MODEL_DIR / "vad_v1.json").read_text())
    gen_postproc(sidecar)
    gen_inference(sidecar)
    gen_e2e(sidecar)
    print(f"wrote fixtures to {TESTDATA}")


if __name__ == "__main__":
    main()
