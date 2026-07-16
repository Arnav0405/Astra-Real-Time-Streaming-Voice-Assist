import numpy as np

from astra_ml.evaluation.segment_eval import label_segments, segment_metrics


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


def test_no_true_segments():
    m = segment_metrics([], [(5, 8)])
    assert m["recall"] is None
    assert m["false_alarms"] == 1
