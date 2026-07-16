from astra_ml.evaluation.tune_postproc import make_grid, pick_best
from astra_ml.postproc import PostprocConfig


def test_make_grid_filters_offset_not_below_onset():
    grid = make_grid(onsets=[0.5, 0.7], offsets=[0.4, 0.6], min_speech=[3], min_silence=[25])
    pairs = {(c.onset_threshold, c.offset_threshold) for c in grid}
    assert pairs == {(0.5, 0.4), (0.7, 0.4), (0.7, 0.6)}  # (0.5, 0.6) dropped
    assert all(isinstance(c, PostprocConfig) for c in grid)


def _result(recall, fa, p90):
    return {
        "segment_recall": recall,
        "false_alarms_per_hour": fa,
        "p90_onset_latency_ms": p90,
    }


def test_pick_best_lowest_fa_among_recall_qualifiers():
    results = [
        ("a", _result(0.99, 80.0, 100.0)),
        ("b", _result(0.96, 20.0, 200.0)),
        ("c", _result(0.90, 5.0, 100.0)),  # recall below floor, excluded
    ]
    assert pick_best(results, min_recall=0.95)[0] == "b"


def test_pick_best_ties_broken_by_latency():
    results = [
        ("slow", _result(0.96, 20.0, 400.0)),
        ("fast", _result(0.96, 20.0, 150.0)),
    ]
    assert pick_best(results, min_recall=0.95)[0] == "fast"


def test_pick_best_falls_back_to_max_recall_when_none_qualify():
    results = [
        ("a", _result(0.80, 5.0, 100.0)),
        ("b", _result(0.90, 50.0, 100.0)),
    ]
    assert pick_best(results, min_recall=0.95)[0] == "b"


def test_pick_best_handles_none_latency():
    results = [
        ("nolat", _result(0.96, 10.0, None)),
        ("lat", _result(0.96, 10.0, 100.0)),
    ]
    # None latency (no matches ever) sorts last
    assert pick_best(results, min_recall=0.95)[0] == "lat"
