"""Dataset-free tests for wake-word eval metrics and tuning."""

import json

import numpy as np
import torch
from test_ww_frontend import fake_embed, fake_melspec

from astra_ml.audio.ww_frontend import DEFAULT_FRONTEND, WwFrontend
from astra_ml.evaluation.tune_ww import pick_best, score_grid, update_sidecar
from astra_ml.evaluation.ww_eval import (
    build_report,
    fa_per_hour,
    recall_and_latency,
    speech_end_sample,
    speech_fa_wavs,
    stream_scores,
    trigger_frames,
)
from astra_ml.postproc_ww import WwPostprocConfig
from astra_ml.training.train_ww import SCORE_STEPS_PER_HOUR

PP = WwPostprocConfig(threshold=0.5, patience_frames=2, refractory_frames=100)


class ConstantHead(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, feats):
        return torch.full((feats.shape[0],), self.value)


def test_stream_scores_cadence():
    fe = WwFrontend(fake_melspec, fake_embed, DEFAULT_FRONTEND)
    audio = np.zeros(1280 * 10 + 700, dtype=np.float32)  # partial tail chunk dropped
    scores = stream_scores(fe, ConstantHead(0.7), audio)
    assert scores.shape == (10,)
    assert np.allclose(scores, 0.7)


def test_trigger_frames_uses_transport_frame_indices():
    scores = np.array([0.9, 0.9, 0.1], dtype=np.float32)
    assert trigger_frames(scores, PP) == [7]  # second chunk ends at frame 7


def test_recall_and_latency():
    hit = np.array([0.1, 0.9, 0.9, 0.1], dtype=np.float32)  # trigger at frame 11
    miss = np.zeros(4, dtype=np.float32)
    recall, latencies, silence_fa = recall_and_latency([hit, miss], [5, 5], PP)
    assert recall == 0.5
    assert latencies == [(11 - 5) * 20.0]
    assert silence_fa == 0


def test_latency_clamped_at_zero():
    hit = np.array([0.9, 0.9], dtype=np.float32)
    _, latencies, _ = recall_and_latency([hit], [50], PP)
    assert latencies == [0.0]


def test_lead_in_trigger_is_a_false_accept_not_a_hit():
    # fires at frame 7, inside the lead-in silence: nothing to detect there
    early = np.array([0.9, 0.9, 0.1, 0.1], dtype=np.float32)
    recall, latencies, silence_fa = recall_and_latency([early], [20], PP, min_frame=10)
    assert recall == 0.0
    assert latencies == []
    assert silence_fa == 1


def test_speech_end_ignores_trailing_silence():
    # a 3 s capture buffer holding a 0.5 s word: word end is 0.5 s, not 3 s
    clip = np.zeros(3 * 16000, dtype=np.float32)
    clip[: 16000 // 2] = 0.8
    assert speech_end_sample(clip) == 16000 // 2


def test_speech_end_of_pure_silence_falls_back_to_clip_length():
    assert speech_end_sample(np.zeros(1000, dtype=np.float32)) == 1000


def test_fa_per_hour():
    # one trigger in one 80 ms chunk-hour: 45000 chunks/hour
    scores = np.zeros(45000, dtype=np.float32)
    scores[100:102] = 0.9
    assert fa_per_hour([scores], PP) == 1.0


def _collected(score: float, n_cat: int = 2):
    """Streams that stay silent through the lead-in, then hold `score`."""
    stream = np.array([0.0] * 15 + [score] * 15, dtype=np.float32)
    cats = ["quiet", "noisy", "test"]
    return {
        "recall": {c: [stream.copy() for _ in range(n_cat)] for c in cats},
        "word_end": {c: [60] * n_cat for c in cats},
        "fa": [],
    }


def test_score_grid_recall_tracks_threshold():
    # 0.6-scoring clips: detected below that threshold, missed above it
    grid = score_grid(_collected(0.6), refractory_frames=100)
    low = [r for r in grid if r["threshold"] == 0.5 and r["patience_frames"] == 2][0]
    high = [r for r in grid if r["threshold"] == 0.9 and r["patience_frames"] == 2][0]
    assert low["recall_quiet"] == 1.0
    assert high["recall_quiet"] == 0.0


def test_pick_best_respects_floors_and_prefers_low_fa():
    results = [
        {
            "threshold": 0.3,
            "patience_frames": 1,
            "recall_quiet": 1.0,
            "recall_noisy": 0.9,
            "fa_per_hour": 5.0,
            "latency_ms_median": 80.0,
        },
        {
            "threshold": 0.6,
            "patience_frames": 2,
            "recall_quiet": 0.96,
            "recall_noisy": 0.85,
            "fa_per_hour": 0.2,
            "latency_ms_median": 160.0,
        },
        {
            "threshold": 0.9,
            "patience_frames": 3,
            "recall_quiet": 0.5,
            "recall_noisy": 0.4,
            "fa_per_hour": 0.0,
            "latency_ms_median": 240.0,
        },
    ]
    best = pick_best(results, floor_quiet=0.95, floor_noisy=0.80)
    assert best["threshold"] == 0.6


def test_pick_best_falls_back_to_max_recall():
    results = [
        {
            "threshold": 0.6,
            "patience_frames": 1,
            "recall_quiet": 0.7,
            "recall_noisy": 0.6,
            "fa_per_hour": 0.1,
            "latency_ms_median": 100.0,
        },
        {
            "threshold": 0.3,
            "patience_frames": 1,
            "recall_quiet": 0.9,
            "recall_noisy": 0.8,
            "fa_per_hour": 2.0,
            "latency_ms_median": 90.0,
        },
    ]
    best = pick_best(results, floor_quiet=0.95, floor_noisy=0.80)
    assert best["threshold"] == 0.3


def _grid_row(threshold, fa, fa_speech):
    return {
        "threshold": threshold,
        "patience_frames": 2,
        "recall_quiet": 1.0,
        "recall_noisy": 0.9,
        "fa_per_hour": fa,
        "fa_per_hour_speech": fa_speech,
        "latency_ms_median": 100.0,
    }


def test_pick_best_prefers_low_speech_fa_over_low_environmental_fa():
    # 0.5 wins on environmental FA but fires constantly on speech, which is the failure
    # mode users actually hit; ordering on environmental FA alone is what shipped it
    results = [_grid_row(0.5, fa=0.1, fa_speech=90.0), _grid_row(0.8, fa=2.0, fa_speech=1.0)]
    assert pick_best(results, floor_quiet=0.95, floor_noisy=0.80)["threshold"] == 0.8


def test_pick_best_falls_back_to_environmental_fa_without_a_speech_set():
    results = [_grid_row(0.5, fa=2.0, fa_speech=None), _grid_row(0.8, fa=0.1, fa_speech=None)]
    assert pick_best(results, floor_quiet=0.95, floor_noisy=0.80)["threshold"] == 0.8


def _collected_for_report(fa_speech_streams):
    high = np.full(20, 0.9, dtype=np.float32)
    return {
        "recall": {"quiet": [high], "noisy": [high], "test": [high]},
        "word_end": {"quiet": [4], "noisy": [4], "test": [4]},
        "fa": [],
        "fa_speech": fa_speech_streams,
    }


def test_speech_gate_fails_when_there_is_no_speech_set(ww_cfg):
    # an absent measurement must not read as a pass — that is how a model that fires on
    # any spoken word cleared the gates
    report = build_report(_collected_for_report([]), PP, ww_cfg)
    assert report["fa_per_hour_speech"] is None
    assert report["gates"]["fa_per_hour_speech"] is False


def test_speech_gate_reflects_the_measured_rate(ww_cfg):
    ww_cfg.eval.max_fa_per_hour_speech = 5.0
    firing = np.full(int(SCORE_STEPS_PER_HOUR), 0.9, dtype=np.float32)  # ~1 h, always over
    assert build_report(_collected_for_report([firing]), PP, ww_cfg)["gates"][
        "fa_per_hour_speech"
    ] is False

    silent = np.zeros(int(SCORE_STEPS_PER_HOUR), dtype=np.float32)
    quiet_report = build_report(_collected_for_report([silent]), PP, ww_cfg)
    assert quiet_report["fa_per_hour_speech"] == 0.0
    assert quiet_report["gates"]["fa_per_hour_speech"] is True


def test_speech_fa_wavs_takes_manifest_rows_and_held_out_negative_sessions(ww_cfg, tmp_path):
    present = tmp_path / "chunk.wav"
    present.write_bytes(b"")
    manifest = tmp_path / "eval_speech.csv"
    manifest.write_text(f"id_a,{present}\nid_b,{tmp_path / 'gone.wav'}\n")
    ww_cfg.eval.fa_speech_manifests = [manifest, tmp_path / "no_such_manifest.csv"]

    root = ww_cfg.data.negative_recordings_root
    root.mkdir(parents=True)
    (root / "manifest_split.csv").write_text(
        "path,session,speaker,env,split\n"
        "session_a/000.wav,a,user,,train\n"
        "session_b/000.wav,b,user,,eval\n"
        "session_c/000.wav,c,user,,test\n"
    )

    got = speech_fa_wavs(ww_cfg)
    # missing manifest rows and a missing manifest are skipped, not fatal
    assert present in got
    assert tmp_path / "gone.wav" not in got
    # held out only: a train session in the FA set would measure FA on trained-on audio
    assert root / "session_b/000.wav" in got
    assert root / "session_c/000.wav" in got
    assert root / "session_a/000.wav" not in got


def test_update_sidecar_touches_only_trigger_knobs(tmp_path):
    sidecar = tmp_path / "ww_v1.json"
    sidecar.write_text(
        json.dumps(
            {
                "sample_rate": 16000,
                "recommended_threshold": 0.5,
                "postproc": {"patience_frames": 1, "refractory_frames": 100},
                "gating": {"preroll_frames": 50},
            }
        )
    )
    update_sidecar(sidecar, {"threshold": 0.65, "patience_frames": 2})
    got = json.loads(sidecar.read_text())
    assert got["recommended_threshold"] == 0.65
    assert got["postproc"]["patience_frames"] == 2
    assert got["postproc"]["refractory_frames"] == 100
    assert got["gating"] == {"preroll_frames": 50}


def test_acav_est_fa_per_hour(ww_cfg):
    import pytest

    from astra_ml.evaluation.ww_eval import acav_est_fa_per_hour
    from astra_ml.models.ww import EMB_DIM, HEAD_FRAMES
    from astra_ml.training.train_ww import SCORE_STEPS_PER_HOUR

    # no ACAV file downloaded yet -> reported as None, never a crash
    assert acav_est_fa_per_hour(ww_cfg, ConstantHead(0.9), 0.5) is None

    np.save(ww_cfg.data.acav_features, np.zeros((20, HEAD_FRAMES, EMB_DIM), dtype=np.float16))
    always = acav_est_fa_per_hour(ww_cfg, ConstantHead(0.9), 0.5)
    assert always == pytest.approx(SCORE_STEPS_PER_HOUR)  # every window fires
    assert acav_est_fa_per_hour(ww_cfg, ConstantHead(0.1), 0.5) == 0.0
