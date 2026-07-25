# RUN: uv run python -m astra_ml.evaluation.ww_eval --config configs/ww_v1.yaml --checkpoint runs/ww/best.pt

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from astra_ml.audio.ww_frontend import DEFAULT_FRONTEND, WwFrontend
from astra_ml.data.oww_assets import load_acav
from astra_ml.data.ww_augment import add_noise, load_mono, scan_wavs_in_folds
from astra_ml.postproc_ww import WwPostprocConfig, WwPostprocessor
from astra_ml.training.train_ww import (
    ACAV_VAL_ROWS,
    INT16_SCALE,
    SCORE_STEPS_PER_HOUR,
    _read_recordings,
    load_head,
)
from astra_ml.training.ww_config import WwConfig, load_ww_config

SR = 16000
CHUNK = 1280
FRAMES_PER_CHUNK = 4  # 20 ms transport frames per score step
LEAD_IN_S = 1.0
TAIL_S = 0.5
NOISY_SNR_DB = 5.0
LEAD_IN_FRAMES = int(LEAD_IN_S * SR) // 320
SPEECH_REL_THRESH = 0.05  # tune knob: fraction of clip peak counted as speech


def speech_end_sample(clip: np.ndarray, rel_thresh: float = SPEECH_REL_THRESH) -> int:
    """Sample index just past the last speech energy in a clip.

    Recordings are fixed-length capture buffers (3 s) with up to ~2 s of trailing
    silence, so len(clip) is the end of the *buffer*, not of the wake word. Using
    the buffer end makes every latency negative and the max(0, ...) clamp reports
    a median of 0 ms regardless of how the model actually behaves.
    """
    # ponytail: relative-energy gate, not the trained VAD — these are close-mic
    # prompted captures. Switch to vad_v1.onnx if room tone ever floats the floor.
    env = np.abs(clip)
    peak = float(env.max()) if len(env) else 0.0
    if peak == 0.0:
        return len(clip)
    above = np.nonzero(env > rel_thresh * peak)[0]
    return int(above[-1]) + 1 if len(above) else len(clip)


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
    min_frame: int = 0,
) -> tuple[float, list[float], int]:
    """Fraction of streams with a trigger + latency (ms after word end) per hit.

    Triggers before `min_frame` land in the prepended lead-in silence, where there
    is nothing to detect. They are false accepts, not hits: counting them inflates
    recall toward 1.0 for a model that simply fires constantly.
    """
    hits, latencies, silence_fa = 0, [], 0
    for scores, end_frame in zip(score_streams, word_end_frames, strict=True):
        triggers = trigger_frames(scores, pp_cfg)
        silence_fa += sum(1 for t in triggers if t < min_frame)
        triggers = [t for t in triggers if t >= min_frame]
        if triggers:
            hits += 1
            latencies.append(max(0.0, (triggers[0] - end_frame) * 20.0))
    recall = hits / len(score_streams) if score_streams else 0.0
    return recall, latencies, silence_fa


def fa_per_hour(score_streams: list[np.ndarray], pp_cfg: WwPostprocConfig) -> float:
    triggers = sum(len(trigger_frames(s, pp_cfg)) for s in score_streams)
    hours = sum(len(s) * CHUNK for s in score_streams) / SR / 3600
    return triggers / hours if hours else 0.0


@torch.no_grad()
def acav_est_fa_per_hour(cfg: WwConfig, head, threshold: float) -> float | None:
    """Windowed FP rate on the ACAV val tail at the shipping threshold, scaled to fa/hr.

    Same held-out rows train_ww validates on, so train and eval numbers are
    comparable. Ignores patience_frames (upper bound). Reported, never gated —
    the fa_per_hour gate stays on real streamed audio.
    """
    if not cfg.data.acav_features.exists():
        return None
    acav = load_acav(cfg.data.acav_features)
    tail = acav[max(0, len(acav) - ACAV_VAL_ROWS) :]
    if not len(tail):
        return None
    fp = 0
    for i in range(0, len(tail), 4096):
        block = torch.from_numpy(np.asarray(tail[i : i + 4096]).astype(np.float32))
        fp += int((head(block) >= threshold).sum())
    return fp / len(tail) * SCORE_STEPS_PER_HOUR


def load_recall_clips(cfg: WwConfig) -> dict[str, list[np.ndarray]]:
    """Category -> clips. Uses recordings when present, else TTS val fallback."""
    split_manifest = cfg.data.recordings_root / "manifest_split.csv"
    out: dict[str, list[np.ndarray]] = {"eval": [], "test": []}
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
            # measure the word end on the clean clip: added noise raises the floor
            end_sample = int(LEAD_IN_S * SR) + speech_end_sample(clip)
            if noisy:
                noise = rng.normal(0, 0.05, len(clip)).astype(np.float32)
                clip = add_noise(clip, noise, NOISY_SNR_DB)
            audio = clip_stream(clip)
            cat_streams.append(stream_scores(frontend, head, audio))
            ends.append(end_sample // 320)  # 20 ms frame index of word end
        streams[name] = cat_streams
        word_end[name] = ends

    category("quiet", clips["eval"], noisy=False)
    category("noisy", clips["eval"], noisy=True)
    category("test", clips["test"], noisy=False)

    fa_streams = []
    for wav in scan_wavs_in_folds(cfg.eval.fa_audio_dirs, cfg.eval.fa_folds):
        fa_streams.append(stream_scores(frontend, head, load_mono(wav)))
    fa_speech_streams = [
        stream_scores(frontend, head, load_mono(wav)) for wav in speech_fa_wavs(cfg)
    ]
    return {
        "recall": streams,
        "word_end": word_end,
        "fa": fa_streams,
        "fa_speech": fa_speech_streams,
    }


def speech_fa_wavs(cfg: WwConfig) -> list[Path]:
    """Speech the model must stay silent through: manifest CSVs + held-out negative sessions.

    fa_audio_dirs cannot carry these. ESC-50 has no speech category and the chime
    speech chunks live interleaved with non-speech ones in a single directory, listed
    only by CSV — an rglob over that directory would sweep in the non-speech chunks
    that fa_audio_dirs already covers.
    """
    wavs: list[Path] = []
    for manifest in cfg.eval.fa_speech_manifests:
        if not manifest.exists():
            print(f"warning: fa_speech_manifest missing, skipped: {manifest}")
            continue
        with open(manifest) as f:
            for row in csv.reader(f):
                if len(row) >= 2 and Path(row[1]).exists():
                    wavs.append(Path(row[1]))
    # eval + test: every negative session training did not see. Both are held out, and
    # a speech FA number wants all the held-out audio it can get.
    root = cfg.data.negative_recordings_root
    for split in ("eval", "test"):
        wavs += _read_recordings(root, split)
    return wavs


def build_report(collected: dict, pp_cfg: WwPostprocConfig, cfg: WwConfig) -> dict:
    report: dict = {"postproc": pp_cfg.__dict__}
    latencies: list[float] = []
    silence_fa = 0
    for name, cat_streams in collected["recall"].items():
        if not cat_streams:
            report[f"recall_{name}"] = None
            continue
        recall, lat, fa = recall_and_latency(
            cat_streams, collected["word_end"][name], pp_cfg, min_frame=LEAD_IN_FRAMES
        )
        report[f"recall_{name}"] = recall
        silence_fa += fa
        if name in ("quiet", "noisy"):
            latencies += lat
    report["lead_in_false_accepts"] = silence_fa
    report["fa_per_hour"] = fa_per_hour(collected["fa"], pp_cfg)
    report["fa_hours"] = sum(len(s) * CHUNK for s in collected["fa"]) / SR / 3600
    speech = collected.get("fa_speech") or []
    report["fa_per_hour_speech"] = fa_per_hour(speech, pp_cfg) if speech else None
    report["fa_hours_speech"] = sum(len(s) * CHUNK for s in speech) / SR / 3600
    report["latency_ms_median"] = float(np.median(latencies)) if latencies else None
    report["latency_ms_p95"] = float(np.percentile(latencies, 95)) if latencies else None

    e = cfg.eval
    report["gates"] = {
        "recall_quiet": report["recall_quiet"] is not None
        and report["recall_quiet"] >= e.recall_floor_quiet,
        "recall_noisy": report["recall_noisy"] is not None
        and report["recall_noisy"] >= e.recall_floor_noisy,
        "fa_per_hour": report["fa_per_hour"] <= e.max_fa_per_hour,
        # Fails when there is no speech FA set at all, unlike the permissive latency gate.
        # An absent measurement is what let a model that fires on any spoken word pass:
        # fa_audio_dirs is environmental audio, so "silent through speech" went unchecked.
        "fa_per_hour_speech": report["fa_per_hour_speech"] is not None
        and report["fa_per_hour_speech"] <= e.max_fa_per_hour_speech,
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
    report["est_fa_per_hour_acav"] = acav_est_fa_per_hour(cfg, head, pp_cfg.threshold)
    out = cfg.training.runs_dir / "ww_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
