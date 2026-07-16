import numpy as np

from astra_ml.evaluation.segment_eval import (
    label_segments,
    latency_rows,
    score_windows,
    segment_metrics,
)
from astra_ml.postproc import PostprocConfig


def test_label_segments_finds_contiguous_runs():
    labels = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1])
    assert label_segments(labels) == [(2, 5), (6, 7), (9, 10)]
    assert label_segments(np.zeros(5)) == []
    assert label_segments(np.ones(4)) == [(0, 4)]


def test_perfect_prediction():
    true = [(10, 20), (40, 50)]
    m = segment_metrics(true, true)
    assert m["recall"] == 1.0
    assert m["false_alarms"] == 0
    assert m["onset_latencies"] == [0, 0]


def test_missed_segment_lowers_recall():
    m = segment_metrics([(10, 20), (40, 50)], [(10, 20)])
    assert m["recall"] == 0.5
    assert m["false_alarms"] == 0


def test_nonoverlapping_prediction_is_false_alarm():
    m = segment_metrics([(10, 20)], [(11, 19), (60, 70)])
    assert m["recall"] == 1.0
    assert m["false_alarms"] == 1


def test_onset_latency_from_first_overlapping_prediction():
    # prediction starts 3 frames into the true segment
    m = segment_metrics([(10, 20)], [(13, 22)])
    assert m["onset_latencies"] == [3]


def test_pre_active_prediction_clamps_latency_to_zero():
    # one pred bridges two utterances; second one must not go negative
    m = segment_metrics([(10, 20), (40, 50)], [(12, 45)])
    assert m["onset_latencies"] == [2, 0]


def test_carried_segment_excluded_from_latency():
    # true segment starting at frame 0 = chopped at window edge, onset unmeasurable
    m = segment_metrics([(0, 20)], [(0, 20)])
    assert m["matched"] == 1
    assert m["onset_latencies"] == []


def test_no_true_segments():
    m = segment_metrics([], [(5, 8)])
    assert m["recall"] is None
    assert m["false_alarms"] == 1


def test_score_windows_aggregates_across_windows():
    cfg = PostprocConfig(
        onset_threshold=0.7, offset_threshold=0.4, min_speech_frames=3, min_silence_frames=5
    )
    # window 1: one true segment, detected exactly
    probs1 = np.full(50, 0.1)
    probs1[10:30] = 0.9
    labels1 = np.zeros(50)
    labels1[10:30] = 1
    # window 2: one true segment missed, one false alarm
    probs2 = np.full(50, 0.1)
    probs2[40:46] = 0.9  # FA: labels say silence
    labels2 = np.zeros(50)
    labels2[5:15] = 1  # missed: probs stay low

    report = score_windows([(probs1, labels1), (probs2, labels2)], cfg)
    assert report["true_segments"] == 2
    assert report["segment_recall"] == 0.5
    assert report["median_onset_latency_ms"] == 0.0
    assert report["measurable_onsets"] == 1
    # 100 frames * 20 ms = 2 s of audio; 1 FA
    assert report["false_alarms_per_hour"] == 1 / (100 * 0.02 / 3600)


def test_latency_rows_one_per_true_segment():
    cfg = PostprocConfig(
        onset_threshold=0.7, offset_threshold=0.4, min_speech_frames=3, min_silence_frames=5
    )
    # window 0: detected 4 frames late
    probs1 = np.full(50, 0.1)
    probs1[14:30] = 0.9
    labels1 = np.zeros(50)
    labels1[10:30] = 1
    # window 1: missed segment + false alarm (FA gets no row)
    probs2 = np.full(50, 0.1)
    probs2[40:46] = 0.9
    labels2 = np.zeros(50)
    labels2[5:15] = 1

    rows = latency_rows([(probs1, labels1), (probs2, labels2)], cfg)
    assert rows == [
        {
            "window": 0,
            "true_start": 10,
            "true_end": 30,
            "pred_start": 14,
            "onset_type": "measured",
            "latency_frames": 4,
            "latency_ms": 80.0,
        },
        {
            "window": 1,
            "true_start": 5,
            "true_end": 15,
            "pred_start": None,
            "onset_type": "missed",
            "latency_frames": None,
            "latency_ms": None,
        },
    ]


def test_latency_rows_classifies_carried_and_pre_active():
    cfg = PostprocConfig(
        onset_threshold=0.7, offset_threshold=0.4, min_speech_frames=3, min_silence_frames=5
    )
    # speech from frame 0 (carried) and a second utterance bridged by hangover
    probs = np.full(50, 0.9)
    probs[20:24] = 0.5  # above offset: machine stays in SPEECH across the gap
    labels = np.ones(50)
    labels[20:24] = 0

    rows = latency_rows([(probs, labels)], cfg)
    assert [r["onset_type"] for r in rows] == ["carried", "pre_active"]
    assert rows[0]["latency_ms"] is None  # unmeasurable
    assert rows[1]["latency_ms"] == 0.0  # detector already active, clamp
