"""Stream-level evaluation and trigger tuning for wake word v2.

    uv run python -m astra_ml.evaluation.kws_eval --config configs/ww_v2.yaml \
        --model ../../assets/models/wakeword/ww_v2.onnx [--tune] [--update-sidecar]

Scores through the *exported ONNX*, not the torch checkpoint. v1 evaluated the torch head
against a separately-frozen frontend, which left room for the shipped graph to differ from
the graded one; here the artifact under test is the artifact that ships.

--tune reuses the same score streams to grid-search threshold x patience, so tuning and
reporting cost one pass over the corpora instead of two. Tuning only ever looks at the
`eval`-split recordings; `recall_test` comes from data.frozen_test_sessions, which is
neither trained on nor tuned against.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from astra_ml.data.kws_dataset import recording_clips, scan_audio, tts_clips
from astra_ml.data.ww_augment import add_noise, load_mono, scan_wavs_in_folds
from astra_ml.evaluation.ww_metrics import (
    CHUNK,
    LEAD_IN_S,
    NOISY_SNR_DB,
    SR,
    build_report,
    clip_stream,
    pick_best,
    score_grid,
    speech_end_sample,
    tune_report,
    update_sidecar,
)
from astra_ml.postproc_ww import WwPostprocConfig
from astra_ml.training.kws_config import KwsConfig, load_kws_config

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL = REPO_ROOT / "assets" / "models" / "wakeword" / "ww_v2.onnx"
DEFAULT_SIDECAR = REPO_ROOT / "assets" / "models" / "wakeword" / "ww_v2.json"
INT16_SCALE = 32768.0
FRAME_SAMPLES = 320


class OnnxScorer:
    """One score per 80 ms chunk, from the exported graph."""

    def __init__(self, model_path: Path, window_samples: int, threads: int = 0) -> None:
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.window_samples = window_samples

    def score(self, window: np.ndarray) -> float:
        out = self.session.run(["prob"], {"audio": window[None, :].astype(np.float32)})[0]
        return float(out[0, 0])


def stream_scores(scorer: OnnxScorer, audio: np.ndarray) -> np.ndarray:
    """Score a float [-1,1] stream at 80 ms cadence with a cold-start window.

    Mirrors the Go detector exactly: zero-initialized window, shift left one chunk,
    append, score (detector.go:30-44).
    """
    window = np.zeros(scorer.window_samples, dtype=np.float32)
    scores = []
    for start in range(0, len(audio) - CHUNK + 1, CHUNK):
        window = np.roll(window, -CHUNK)
        window[-CHUNK:] = audio[start : start + CHUNK] * INT16_SCALE
        scores.append(scorer.score(window))
    return np.asarray(scores, dtype=np.float32)


def load_recall_clips(cfg: KwsConfig) -> dict[str, list[np.ndarray]]:
    """Category -> clips. Uses recordings when present, else TTS val fallback."""
    out = {
        "eval": [load_mono(p) for p in recording_clips(cfg.data.recordings_root, ("eval",))],
        "test": [load_mono(p) for p in recording_clips(cfg.data.recordings_root, ("test",))],
    }
    if not out["eval"] and not out["test"]:
        print("warning: no recordings; falling back to TTS val clips (dev-only numbers)")
        out["eval"] = [load_mono(p) for p in tts_clips(cfg.data.tts_out, "positive", "val")]
    return out


def speech_fa_wavs(cfg: KwsConfig) -> list[Path]:
    """Speech the model must stay silent through.

    Three sources, all held out: LibriSpeech test-clean (speaker-disjoint from the
    train-clean-100 negatives training used), the chime speech manifests, and every
    negative recording session training did not see.
    """
    wavs = scan_audio(cfg.eval.fa_speech_dirs)
    for manifest in cfg.eval.fa_speech_manifests:
        if not manifest.exists():
            print(f"warning: fa_speech_manifest missing, skipped: {manifest}")
            continue
        with open(manifest) as f:
            wavs += [
                Path(row[1]) for row in csv.reader(f) if len(row) >= 2 and Path(row[1]).exists()
            ]
    for split in ("eval", "test"):
        wavs += recording_clips(cfg.data.negative_recordings_root, (split,))
    return wavs


def collect_streams(cfg: KwsConfig, scorer: OnnxScorer) -> dict:
    """All score streams the report and the tuning grid need, computed once."""
    clips = load_recall_clips(cfg)
    rng = np.random.default_rng(0)
    streams: dict[str, list[np.ndarray]] = {}
    word_end: dict[str, list[int]] = {}

    def category(name: str, raw_clips: list[np.ndarray], noisy: bool) -> None:
        cat_streams, ends = [], []
        for clip in raw_clips:
            # measure the word end on the clean clip: added noise raises the floor
            end_sample = int(LEAD_IN_S * SR) + speech_end_sample(clip)
            if noisy:
                noise = rng.normal(0, 0.05, len(clip)).astype(np.float32)
                clip = add_noise(clip, noise, NOISY_SNR_DB)
            cat_streams.append(stream_scores(scorer, clip_stream(clip)))
            ends.append(end_sample // FRAME_SAMPLES)
        streams[name] = cat_streams
        word_end[name] = ends

    category("quiet", clips["eval"], noisy=False)
    category("noisy", clips["eval"], noisy=True)
    category("test", clips["test"], noisy=False)

    fa = [
        stream_scores(scorer, load_mono(w))
        for w in scan_wavs_in_folds(cfg.eval.fa_audio_dirs, cfg.eval.fa_folds)
    ]
    fa_speech = [stream_scores(scorer, load_mono(w)) for w in speech_fa_wavs(cfg)]
    return {"recall": streams, "word_end": word_end, "fa": fa, "fa_speech": fa_speech}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v2.yaml"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--tune", action="store_true", help="also grid-search threshold x patience")
    parser.add_argument("--update-sidecar", action="store_true", help="implies --tune")
    args = parser.parse_args()

    cfg = load_kws_config(args.config)
    refractory_frames = int(cfg.postproc.refractory_seconds * SR / FRAME_SAMPLES)
    scorer = OnnxScorer(args.model, cfg.frontend.window_samples)
    collected = collect_streams(cfg, scorer)
    runs_dir = cfg.training.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)

    threshold, patience = cfg.postproc.threshold, cfg.postproc.patience_frames
    if args.tune or args.update_sidecar:
        results = score_grid(collected, refractory_frames)
        best = pick_best(results, cfg.eval.recall_floor_quiet, cfg.eval.recall_floor_noisy)
        report = tune_report(results, best, refractory_frames, cfg)
        (runs_dir / "tune_report.json").write_text(json.dumps(report, indent=2) + "\n")
        threshold, patience = best["threshold"], best["patience_frames"]
        print(f"tuned: threshold {threshold}, patience {patience}")
        if args.update_sidecar:
            update_sidecar(args.sidecar, best)
            print(f"sidecar updated: {args.sidecar}")

    pp_cfg = WwPostprocConfig(threshold, patience, refractory_frames)
    report = build_report(collected, pp_cfg, cfg)
    out = runs_dir / "ww_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report: {out}")
    if not all(report["gates"].values()):
        failed = [k for k, v in report["gates"].items() if not v]
        print(f"GATES FAILED: {failed}")


if __name__ == "__main__":
    main()
