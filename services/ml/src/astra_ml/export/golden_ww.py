"""Golden fixtures binding the Python wake-word reference to the Go port.

Writes into services/backend/internal/wakeword/testdata/:

- ww_postproc_golden.json   trigger machine parity (exact)
- ww_gating_golden.json     sink gating rules: backfill, 4-frame chunking,
                            partial-chunk drop, reset, refractory across gates.
                            Model-free: scores are scripted for a fake scorer;
                            chunks are pinned by checksum of the PCM fed.
- ww_inference_golden.json  merged-model scores over deterministic PCM
                            (needs the committed assets/models/wakeword
                            artifacts; skipped when absent)

The first two fixtures are model-free and shared by every model version. Only the
inference and e2e fixtures are tied to a specific graph, so --name selects which one:
ww_v1 writes the unsuffixed files, anything else appends its own suffix
(ww_v2 -> ww_inference_golden_v2.json), letting both models be graded side by side.

GatingSim below IS the reference semantics for wakeword.Sink:

- per incoming 20 ms frame: buffer it; deliver any scripted VAD event for this
  frame index; else if the gate is open, feed the frame to the detector
- on speech_start (retroactive ev.frame): open the gate and backfill from
  max(0, ev.frame - preroll_frames) through the current frame inclusive
- feeding: every 4th fed frame closes a 1280-sample chunk -> score -> postproc
  push with the absolute frame index of the chunk's last frame
- on speech_end: the current frame is NOT fed, the partial chunk is dropped,
  the score window resets, postproc.gate_reset() (refractory clock survives)

Run from services/ml: uv run python -m astra_ml.export.golden_ww
"""

import base64
import json
from pathlib import Path

import numpy as np

from astra_ml.postproc_ww import WwPostprocConfig, WwPostprocessor

REPO_ROOT = Path(__file__).resolve().parents[5]
MODEL_DIR = REPO_ROOT / "assets" / "models" / "wakeword"
TESTDATA = REPO_ROOT / "services" / "backend" / "internal" / "wakeword" / "testdata"

FRAME_SAMPLES = 320
FRAMES_PER_CHUNK = 4
CHUNK_SAMPLES = FRAME_SAMPLES * FRAMES_PER_CHUNK

PP = WwPostprocConfig(threshold=0.5, patience_frames=2, refractory_frames=100)


# ---------------------------------------------------------------- postproc


def postproc_steps() -> list[dict]:
    """Score/frame sequence hitting every branch of the trigger machine."""
    steps = []
    frame = -1

    def push(scores, gate_reset_first=False):
        nonlocal frame
        for i, s in enumerate(scores):
            frame += FRAMES_PER_CHUNK
            steps.append(
                {"score": s, "frame": frame, "gate_reset_before": gate_reset_first and i == 0}
            )

    push([0.9, 0.1])  # single blip below patience
    push([0.9, 0.9, 0.9])  # trigger at 2nd step; 3rd suppressed by refractory
    push([0.4, 0.9], gate_reset_first=True)  # gate cycle; run restarts
    push([0.9] * 30)  # sustained: next trigger only after refractory expires
    push([0.5, 0.5], gate_reset_first=True)  # threshold inclusive
    return steps


def gen_postproc() -> None:
    steps = postproc_steps()
    pp = WwPostprocessor(PP)
    triggers = []
    for s in steps:
        if s["gate_reset_before"]:
            pp.gate_reset()
        t = pp.push(s["score"], s["frame"])
        if t is not None:
            triggers.append(t)
    out = {
        "postproc": PP.__dict__,
        "steps": steps,
        "triggers": triggers,
    }
    _write("ww_postproc_golden.json", out)


# ---------------------------------------------------------------- gating


class GatingSim:
    """Reference for wakeword.Sink gating. score_fn(chunk_pcm, k) -> score."""

    def __init__(self, pp_cfg: WwPostprocConfig, preroll_frames: int, score_fn):
        self.pp = WwPostprocessor(pp_cfg)
        self.preroll = preroll_frames
        self.score_fn = score_fn
        self.frames: list[np.ndarray] = []  # all seen frames (ring buffer stand-in)
        self.gate_open = False
        self.pending: list[int] = []  # fed frame indices not yet forming a chunk
        self.chunk_count = 0
        self.chunks: list[dict] = []
        self.wakes: list[int] = []

    def _feed(self, idx: int) -> None:
        self.pending.append(idx)
        if len(self.pending) < FRAMES_PER_CHUNK:
            return
        pcm = np.concatenate([self.frames[i] for i in self.pending])
        checksum = int(pcm.astype(np.int64).sum() % 2**31)
        score = self.score_fn(pcm, self.chunk_count)
        self.chunk_count += 1
        last = self.pending[-1]
        self.chunks.append({"frame": last, "checksum": checksum, "score": score})
        self.pending = []
        t = self.pp.push(score, last)
        if t is not None:
            self.wakes.append(t)

    def push_frame(self, idx: int, pcm: np.ndarray, event: dict | None) -> None:
        self.frames.append(pcm)
        if event is not None and event["type"] == "start":
            self.gate_open = True
            for i in range(max(0, event["frame"] - self.preroll), idx + 1):
                self._feed(i)
        elif event is not None and event["type"] == "end":
            self.gate_open = False
            self.pending = []  # partial chunk dropped; engine window resets too
            self.pp.gate_reset()
        elif self.gate_open:
            self._feed(idx)


def gen_gating() -> None:
    rng = np.random.default_rng(0)
    n_frames = 400
    pcm = (rng.uniform(-0.2, 0.2, n_frames * FRAME_SAMPLES) * 32767).astype(np.int16)
    frames = [pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES] for i in range(n_frames)]

    events = {
        60: {"type": "start", "frame": 55},
        120: {"type": "end", "frame": 116},
        140: {"type": "start", "frame": 136},
        180: {"type": "end", "frame": 176},
        260: {"type": "start", "frame": 255},
        340: {"type": "end", "frame": 336},
    }
    # scripted scores per chunk index: gate 1 wakes; gate 2 is high but inside
    # refractory; gate 3 wakes again after refractory expires
    high_at = {14, 15, 20, 21, 40, 41, 42, 60, 61}
    scores = [0.9 if k in high_at else 0.1 for k in range(200)]

    preroll = 50
    sim = GatingSim(PP, preroll, lambda _pcm, k: scores[k])
    for i in range(n_frames):
        sim.push_frame(i, frames[i], events.get(i))

    out = {
        "postproc": PP.__dict__,
        "gating": {"preroll_frames": preroll, "partial_chunk": "drop"},
        "n_frames": n_frames,
        "pcm_s16le_base64": base64.b64encode(pcm.tobytes()).decode(),
        "vad_events": [{"deliver_at": k, **v} for k, v in sorted(events.items())],
        "scores": scores[: sim.chunk_count],
        "expected_chunks": sim.chunks,
        "wake_frames": sim.wakes,
    }
    assert sim.wakes, "gating fixture must contain at least one wake"
    _write("ww_gating_golden.json", out)


# ---------------------------------------------------------------- inference


def _suffix(name: str) -> str:
    """ww_v1 keeps the original filenames; later versions get their own."""
    return "" if name == "ww_v1" else name.removeprefix("ww")


def gen_inference(name: str = "ww_v1") -> None:
    model_path = MODEL_DIR / f"{name}.onnx"
    sidecar_path = MODEL_DIR / f"{name}.json"
    if not model_path.exists() or not sidecar_path.exists():
        print(f"ww_inference_golden{_suffix(name)}: skipped (no committed {name} artifacts yet)")
        return
    import onnxruntime as ort

    sidecar = json.loads(sidecar_path.read_text())
    window_samples = sidecar["window_samples"]
    io = sidecar["io"]["merged"]

    rng = np.random.default_rng(0)
    n_chunks = 40
    pcm = (rng.uniform(-0.5, 0.5, n_chunks * CHUNK_SAMPLES) * 32767).astype(np.int16)

    session = ort.InferenceSession(str(model_path))
    window = np.zeros(window_samples, dtype=np.float32)
    scores = []
    for k in range(n_chunks):
        chunk = pcm[k * CHUNK_SAMPLES : (k + 1) * CHUNK_SAMPLES].astype(np.float32)
        window = np.roll(window, -CHUNK_SAMPLES)
        window[-CHUNK_SAMPLES:] = chunk
        prob = session.run(None, {io["input"]: window[None]})[0]
        scores.append(float(prob.reshape(-1)[0]))

    out = {
        "window_samples": window_samples,
        "chunk_samples": CHUNK_SAMPLES,
        "pcm_s16le_base64": base64.b64encode(pcm.tobytes()).decode(),
        "scores": scores,
    }
    _write(f"ww_inference_golden{_suffix(name)}.json", out)


def gen_e2e(name: str = "ww_v1") -> None:
    """Full-stack fixture: a real recorded "Astraa" clip + the vad and wake
    events the runtime must produce. Needs the committed vad + ww artifacts and
    a recordings manifest; skipped until they exist."""
    import csv

    ww_model = MODEL_DIR / f"{name}.onnx"
    ww_sidecar_path = MODEL_DIR / f"{name}.json"
    vad_model = REPO_ROOT / "assets" / "models" / "vad" / "vad_v1.onnx"
    vad_sidecar_path = REPO_ROOT / "assets" / "models" / "vad" / "vad_v1.json"
    recordings = Path("datasets/ww_recordings")
    manifest = recordings / "manifest_split.csv"
    needed = [ww_model, ww_sidecar_path, vad_model, vad_sidecar_path, manifest]
    if not all(p.exists() for p in needed):
        print("e2e_ww_golden: skipped (missing artifacts or recordings)")
        return

    import onnxruntime as ort
    import soundfile as sf

    from astra_ml.postproc import PostprocConfig, VadPostprocessor

    with open(manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if not rows:
        print("e2e_ww_golden: skipped (no test-split recordings)")
        return
    clip, sr = sf.read(recordings / rows[0]["path"], dtype="float32")
    assert sr == 16000
    lead, tail = np.zeros(16000, dtype=np.float32), np.zeros(8000, dtype=np.float32)
    audio = np.concatenate([lead, clip, tail])
    n_frames = len(audio) // FRAME_SAMPLES
    audio = audio[: n_frames * FRAME_SAMPLES]
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    # VAD pass: streaming GRU model + reference postproc, recording both the
    # retroactive event frame and the frame the event was delivered at.
    vad_sidecar = json.loads(vad_sidecar_path.read_text())
    vad_session = ort.InferenceSession(str(vad_model))
    state = np.zeros(vad_sidecar["state_shape"], dtype=np.float32)
    vpp = VadPostprocessor(PostprocConfig.from_sidecar(vad_sidecar))
    events = {}
    vad_events = []
    for i in range(n_frames):
        frame = pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES].astype(np.float32) / 32768.0
        prob, state = vad_session.run(None, {"pcm": frame[None], "state_in": state})
        ev = vpp.push(float(prob.reshape(-1)[0]))
        if ev is not None:
            events[i] = {"type": ev[0], "frame": ev[1]}
            vad_events.append(list(ev))
    vad_events.extend(list(ev) for ev in vpp.finish())

    # Wake pass: real merged model behind the gating reference.
    ww_sidecar = json.loads(ww_sidecar_path.read_text())
    ww_session = ort.InferenceSession(str(ww_model))
    window_samples = ww_sidecar["window_samples"]
    io = ww_sidecar["io"]["merged"]
    pp_cfg = WwPostprocConfig.from_sidecar(ww_sidecar)

    window = np.zeros(window_samples, dtype=np.float32)

    def score_fn(chunk_pcm: np.ndarray, _k: int) -> float:
        nonlocal window
        window = np.roll(window, -len(chunk_pcm))
        window[-len(chunk_pcm) :] = chunk_pcm.astype(np.float32)
        return float(ww_session.run(None, {io["input"]: window[None]})[0].reshape(-1)[0])

    sim = GatingSim(pp_cfg, ww_sidecar["gating"]["preroll_frames"], score_fn)
    frames = [pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES] for i in range(n_frames)]
    for i in range(n_frames):
        ev = events.get(i)
        if ev is not None and ev["type"] == "end":
            window[:] = 0.0  # engine window reset on gate close
        sim.push_frame(i, frames[i], ev)

    out = {
        "source": rows[0]["path"],
        "pcm_s16le_base64": base64.b64encode(pcm.tobytes()).decode(),
        "vad_events": vad_events,
        "wake_frames": sim.wakes,
    }
    e2e_dir = REPO_ROOT / "services" / "backend" / "tests" / "testdata"
    e2e_dir.mkdir(parents=True, exist_ok=True)
    path = e2e_dir / f"e2e_ww_golden{_suffix(name)}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path} (wakes: {sim.wakes})")


def _write(name: str, obj: dict) -> None:
    TESTDATA.mkdir(parents=True, exist_ok=True)
    path = TESTDATA / name
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"wrote {path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="ww_v1", help="model basename in assets/models/wakeword")
    args = parser.parse_args()

    # Model-free, so generated once regardless of --name.
    gen_postproc()
    gen_gating()
    gen_inference(args.name)
    gen_e2e(args.name)


if __name__ == "__main__":
    main()
