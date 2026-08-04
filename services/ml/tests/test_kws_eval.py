"""Stream-level eval checks for wake-word v2, against a real exported ONNX graph."""

import dataclasses

import numpy as np
import pytest
from test_kws_dataset import _cfg, corpus  # noqa: F401  (fixture reuse)

from astra_ml.evaluation.kws_eval import (
    OnnxScorer,
    collect_streams,
    load_recall_clips,
    speech_fa_wavs,
    stream_scores,
)
from astra_ml.evaluation.ww_metrics import CHUNK, build_report, pick_best, score_grid
from astra_ml.export.export_kws import export
from astra_ml.models.bcresnet import BCResNets
from astra_ml.postproc_ww import WwPostprocConfig


@pytest.fixture
def scored(corpus, tmp_path):  # noqa: F811
    """A cfg plus a scorer over a freshly exported (untrained) graph."""
    cfg = _cfg(corpus)
    cfg = dataclasses.replace(
        cfg,
        eval=dataclasses.replace(cfg.eval, fa_speech_dirs=[corpus / "librispeech" / "test-clean"]),
    )
    torch_model = BCResNets(base_c=cfg.model.base_c, num_classes=1).eval()
    path = export(cfg, torch_model, tmp_path, threshold=0.85, patience=2)
    return cfg, OnnxScorer(path, cfg.frontend.window_samples)


def test_stream_scores_runs_at_the_80ms_cadence(scored):
    _, scorer = scored
    audio = np.zeros(CHUNK * 7 + 100, dtype=np.float32)
    scores = stream_scores(scorer, audio)
    assert scores.shape == (7,)  # the trailing partial chunk is dropped
    assert scores.dtype == np.float32
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_stream_scores_starts_from_a_zero_window(scored):
    """Matches detector.go: the window is zeroed at cold start, so the first score
    sees mostly silence rather than a repeat of the first chunk."""
    _, scorer = scored
    audio = np.concatenate([np.zeros(CHUNK, np.float32), np.ones(CHUNK, np.float32) * 0.3])
    first, second = stream_scores(scorer, audio)
    assert first != second


def test_recall_clips_come_from_the_recording_splits(scored, corpus):  # noqa: F811
    cfg, _ = scored
    clips = load_recall_clips(cfg)
    # The toy corpus has one 'eval' session and no 'test' session.
    assert len(clips["eval"]) == 1
    assert clips["test"] == []


def test_speech_fa_set_includes_librispeech_and_held_out_negatives(scored):
    cfg, _ = scored
    wavs = speech_fa_wavs(cfg)
    names = {p.parent.parent.name for p in wavs}
    assert "test-clean" in names  # LibriSpeech, speaker-disjoint from training
    assert any(p.name == "000.wav" and "n2" in p.parts for p in wavs)  # held-out neg session


def test_collect_streams_fills_every_category(scored):
    cfg, scorer = scored
    collected = collect_streams(cfg, scorer)
    assert set(collected) == {"recall", "word_end", "fa", "fa_speech"}
    assert len(collected["recall"]["quiet"]) == 1
    assert len(collected["recall"]["noisy"]) == 1
    assert collected["fa"], "ESC-50 fold 5 should produce a false-accept stream"
    assert collected["fa_speech"], "LibriSpeech test-clean should produce a speech stream"
    for name, streams in collected["recall"].items():
        assert len(streams) == len(collected["word_end"][name])


def test_report_carries_every_gate(scored):
    cfg, scorer = scored
    collected = collect_streams(cfg, scorer)
    report = build_report(collected, WwPostprocConfig(0.85, 2, 100), cfg)
    assert set(report["gates"]) == {
        "recall_quiet",
        "recall_noisy",
        "fa_per_hour",
        "fa_per_hour_speech",
        "latency",
    }
    assert report["fa_hours"] > 0
    assert report["fa_hours_speech"] > 0


def test_tuning_grid_covers_thresholds_and_patiences(scored):
    cfg, scorer = scored
    collected = collect_streams(cfg, scorer)
    results = score_grid(collected, refractory_frames=100)
    assert len(results) == 14 * 3  # thresholds 0.30..0.95 step 0.05 x patience 1,2,3
    best = pick_best(results, cfg.eval.recall_floor_quiet, cfg.eval.recall_floor_noisy)
    assert best["threshold"] in {r["threshold"] for r in results}
    assert best["patience_frames"] in (1, 2, 3)
