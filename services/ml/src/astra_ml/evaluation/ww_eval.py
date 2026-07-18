"""Wake-word evaluation: recall, false accepts per hour, detection latency.

Replays audio through the full detection pipeline exactly as the Go runtime
will run it: zero-initialized score window, 1280-sample chunk cadence, head
score per chunk, WwPostprocessor triggers. Recall streams are built as
lead-in silence + clip + tail so utterance-initial wake words are scored with
the same cold-start window the runtime sees at VAD gate open.

Categories:
- recall_quiet / recall_noisy: user "eval"-split recordings, as-is vs mixed
  with deterministic noise at 5 dB SNR
- recall_test: frozen test sessions (the gate metric; never tuned against)
- recall_family: eval_only sessions (reported, not gating)
- fa_per_hour: triggers over eval.fa_audio_dirs streams, ungated (an honest
  upper bound — VAD gating in production only removes candidates)

Falls back to TTS val clips when no recordings exist (early dev only).

    uv run python -m astra_ml.evaluation.ww_eval --config configs/ww_v1.yaml \
        --checkpoint runs/ww/best.pt
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from astra_ml.audio.ww_frontend import DEFAULT_FRONTEND, WwFrontend
from astra_ml.data.ww_augment import add_noise, load_mono, scan_wavs
from astra_ml.postproc_ww import WwPostprocConfig, WwPostprocessor
from astra_ml.training.train_ww import INT16_SCALE, load_head
from astra_ml.training.ww_config import WwConfig, load_ww_config

SR = 16000
CHUNK = 1280
FRAMES_PER_CHUNK = 4  # 20 ms transport frames per score step
LEAD_IN_S = 1.0
TAIL_S = 0.5
NOISY_SNR_DB = 5.0


def stream_scores(frontend: WwFrontend, head, audio: np.ndarray) -> np.ndarray:
    """Score a float [-1,1] stream at 80 ms cadence with a cold-start window."""
    window = np.zeros(frontend.cfg.window_samples, dtype=np.float32)
    scores = []
    with torch.no_grad():
        for start in range(0, len(audio) - CHUNK + 1, CHUNK):
            window = np.roll(window, -CHUNK)
            window[-CHUNK:] = audio[start : start + CHUNK] * INT16_SCALE
            feats = frontend.features(window)
            score = head(torch.from_numpy(feats.astype(np.float32))[None]).item()
            scores.append(score)
    return np.asarray(scores, dtype=np.float32)


def trigger_frames(scores: np.ndarray, pp_cfg: WwPostprocConfig) -> list[int]:
    pp = WwPostprocessor(pp_cfg)
    out = []
    for i, score in enumerate(scores):
        frame = (i + 1) * FRAMES_PER_CHUNK - 1
        t = pp.push(float(score), frame)
        if t is not None:
            out.append(t)
    return out


def clip_stream(clip: np.ndarray) -> np.ndarray:
    lead = np.zeros(int(LEAD_IN_S * SR), dtype=np.float32)
    tail = np.zeros(int(TAIL_S * SR), dtype=np.float32)
    return np.concatenate([lead, clip, tail])


def recall_and_latency(
    score_streams: list[np.ndarray],
    word_end_frames: list[int],
    pp_cfg: WwPostprocConfig,
) -> tuple[float, list[float]]:
    """Fraction of streams with a trigger + latency (ms after word end) per hit."""
    hits, latencies = 0, []
    for scores, end_frame in zip(score_streams, word_end_frames, strict=True):
        triggers = trigger_frames(scores, pp_cfg)
        if triggers:
            hits += 1
            latencies.append(max(0.0, (triggers[0] - end_frame) * 20.0))
    recall = hits / len(score_streams) if score_streams else 0.0
    return recall, latencies


def fa_per_hour(score_streams: list[np.ndarray], pp_cfg: WwPostprocConfig) -> float:
    triggers = sum(len(trigger_frames(s, pp_cfg)) for s in score_streams)
    hours = sum(len(s) * CHUNK for s in score_streams) / SR / 3600
    return triggers / hours if hours else 0.0


def load_recall_clips(cfg: WwConfig) -> dict[str, list[np.ndarray]]:
    """Category -> clips. Uses recordings when present, else TTS val fallback."""
    split_manifest = cfg.data.recordings_root / "manifest_split.csv"
    out: dict[str, list[np.ndarray]] = {"eval": [], "test": [], "eval_only": []}
    if split_manifest.exists():
        with open(split_manifest) as f:
            for row in csv.DictReader(f):
                if row["split"] in out:
                    out[row["split"]].append(load_mono(cfg.data.recordings_root / row["path"]))
        return out

    print("warning: no recordings; falling back to TTS val clips (dev-only numbers)")
    with open(cfg.data.tts_out / "manifest.csv") as f:
        for row in csv.DictReader(f):
            if row["label"] == "positive" and row["split"] == "val":
                out["eval"].append(load_mono(cfg.data.tts_out / row["path"]))
    return out


def collect_streams(cfg: WwConfig, frontend: WwFrontend, head) -> dict:
    """All score streams the report (and tune_ww) needs, computed once."""
    clips = load_recall_clips(cfg)
    rng = np.random.default_rng(0)
    word_end = {}
    streams: dict[str, list[np.ndarray]] = {}

    def category(name: str, raw_clips: list[np.ndarray], noisy: bool) -> None:
        cat_streams, ends = [], []
        for clip in raw_clips:
            if noisy:
                noise = rng.normal(0, 0.05, len(clip)).astype(np.float32)
                clip = add_noise(clip, noise, NOISY_SNR_DB)
            audio = clip_stream(clip)
            cat_streams.append(stream_scores(frontend, head, audio))
            end_sample = int(LEAD_IN_S * SR) + len(clip)
            ends.append(end_sample // 320)  # 20 ms frame index of word end
        streams[name] = cat_streams
        word_end[name] = ends

    category("quiet", clips["eval"], noisy=False)
    category("noisy", clips["eval"], noisy=True)
    category("test", clips["test"], noisy=False)
    category("family", clips["eval_only"], noisy=False)

    fa_streams = []
    for wav in scan_wavs(cfg.eval.fa_audio_dirs):
        fa_streams.append(stream_scores(frontend, head, load_mono(wav)))
    return {"recall": streams, "word_end": word_end, "fa": fa_streams}


def build_report(collected: dict, pp_cfg: WwPostprocConfig, cfg: WwConfig) -> dict:
    report: dict = {"postproc": pp_cfg.__dict__}
    latencies: list[float] = []
    for name, cat_streams in collected["recall"].items():
        if not cat_streams:
            report[f"recall_{name}"] = None
            continue
        recall, lat = recall_and_latency(cat_streams, collected["word_end"][name], pp_cfg)
        report[f"recall_{name}"] = recall
        if name in ("quiet", "noisy"):
            latencies += lat
    report["fa_per_hour"] = fa_per_hour(collected["fa"], pp_cfg)
    report["fa_hours"] = sum(len(s) * CHUNK for s in collected["fa"]) / SR / 3600
    report["latency_ms_median"] = float(np.median(latencies)) if latencies else None
    report["latency_ms_p95"] = float(np.percentile(latencies, 95)) if latencies else None

    e = cfg.eval
    report["gates"] = {
        "recall_quiet": report["recall_quiet"] is not None
        and report["recall_quiet"] >= e.recall_floor_quiet,
        "recall_noisy": report["recall_noisy"] is not None
        and report["recall_noisy"] >= e.recall_floor_noisy,
        "fa_per_hour": report["fa_per_hour"] <= e.max_fa_per_hour,
        "latency": report["latency_ms_median"] is None
        or report["latency_ms_median"] <= e.max_latency_ms,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v1.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/ww/best.pt"))
    args = parser.parse_args()
    cfg = load_ww_config(args.config)

    frontend = WwFrontend.from_onnx(cfg.data.frontends_dir, DEFAULT_FRONTEND)
    head = load_head(args.checkpoint)
    pp_cfg = WwPostprocConfig(
        threshold=cfg.postproc.threshold,
        patience_frames=cfg.postproc.patience_frames,
        refractory_frames=int(cfg.postproc.refractory_seconds * SR / 320),
    )

    collected = collect_streams(cfg, frontend, head)
    report = build_report(collected, pp_cfg, cfg)
    out = cfg.training.runs_dir / "ww_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
