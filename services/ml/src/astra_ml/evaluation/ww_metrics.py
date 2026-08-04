"""Model-agnostic wake-word metrics: triggers, recall/latency, FA per hour, tuning grid.

Everything here operates on score streams — arrays of per-80 ms-chunk probabilities — so
it is shared unchanged between v1 (frozen OWW frontend + MLP head) and v2 (BC-ResNet).
Moved out of ww_eval/tune_ww so the v2 path does not have to import the openwakeword
dependency chain to compute a false-accept rate.
"""

import json
from pathlib import Path

import numpy as np

from astra_ml.postproc_ww import WwPostprocConfig, WwPostprocessor

SR = 16000
CHUNK = 1280
FRAMES_PER_CHUNK = 4  # 20 ms transport frames per score step
LEAD_IN_S = 1.0
TAIL_S = 0.5
NOISY_SNR_DB = 5.0
LEAD_IN_FRAMES = int(LEAD_IN_S * SR) // 320
SPEECH_REL_THRESH = 0.05  # tune knob: fraction of clip peak counted as speech

THRESHOLDS = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
PATIENCES = [1, 2, 3]


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


def build_report(collected: dict, pp_cfg: WwPostprocConfig, cfg) -> dict:
    """`cfg` is any config exposing .eval with the gate floors — v1 and v2 both do."""
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


def score_grid(collected: dict, refractory_frames: int) -> list[dict]:
    results = []
    for threshold in THRESHOLDS:
        for patience in PATIENCES:
            pp_cfg = WwPostprocConfig(threshold, patience, refractory_frames)
            recall_q, lat_q, _ = recall_and_latency(
                collected["recall"]["quiet"],
                collected["word_end"]["quiet"],
                pp_cfg,
                min_frame=LEAD_IN_FRAMES,
            )
            recall_n, lat_n, _ = recall_and_latency(
                collected["recall"]["noisy"],
                collected["word_end"]["noisy"],
                pp_cfg,
                min_frame=LEAD_IN_FRAMES,
            )
            latencies = lat_q + lat_n
            results.append(
                {
                    "threshold": threshold,
                    "patience_frames": patience,
                    "recall_quiet": recall_q,
                    "recall_noisy": recall_n,
                    "fa_per_hour": fa_per_hour(collected["fa"], pp_cfg),
                    "fa_per_hour_speech": (
                        fa_per_hour(collected["fa_speech"], pp_cfg)
                        if collected.get("fa_speech")
                        else None
                    ),
                    "latency_ms_median": float(np.median(latencies)) if latencies else None,
                }
            )
    return results


def _inf_if_none(value: float | None) -> float:
    return float("inf") if value is None else value


def pick_best(results: list[dict], floor_quiet: float, floor_noisy: float) -> dict:
    ok = [
        r for r in results if r["recall_quiet"] >= floor_quiet and r["recall_noisy"] >= floor_noisy
    ]
    if not ok:
        # nothing meets the floors: maximize recall instead so the report is useful
        return max(results, key=lambda r: r["recall_quiet"] + r["recall_noisy"])
    # Speech FA leads: minimizing environmental FA alone is what picked a threshold that
    # fires on any spoken word. With no speech set every key is inf and this degrades to
    # the old environmental-FA ordering.
    return min(
        ok,
        key=lambda r: (
            _inf_if_none(r.get("fa_per_hour_speech")),
            r["fa_per_hour"],
            _inf_if_none(r["latency_ms_median"]),
        ),
    )


def tune_report(results: list[dict], best: dict, refractory_frames: int, cfg) -> dict:
    return {
        "best": {**best, "refractory_frames": refractory_frames},
        "floors": {
            "recall_quiet": cfg.eval.recall_floor_quiet,
            "recall_noisy": cfg.eval.recall_floor_noisy,
        },
        "combos_scored": len(results),
        # NOT a ranking — lowest-FA combos, which are the most conservative and worst-recall
        # on the grid. The setting to ship is "best" above; this list is for seeing the curve.
        "lowest_fa_combos_diagnostic_only": sorted(
            results, key=lambda r: (_inf_if_none(r.get("fa_per_hour_speech")), r["fa_per_hour"])
        )[:10],
    }


def update_sidecar(sidecar_path: Path, best: dict) -> None:
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["recommended_threshold"] = best["threshold"]
    sidecar["postproc"]["patience_frames"] = best["patience_frames"]
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
