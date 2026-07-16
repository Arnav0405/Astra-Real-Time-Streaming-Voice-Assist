import numpy as np

from astra_ml.postproc import PostprocConfig, VadPostprocessor, segments

CFG = PostprocConfig(
    onset_threshold=0.7,
    offset_threshold=0.4,
    min_speech_frames=3,
    min_silence_frames=5,
)


def test_silence_yields_no_segments():
    assert segments(np.full(50, 0.1), CFG) == []


def test_clean_burst_yields_one_segment_at_run_start():
    probs = np.full(30, 0.1)
    probs[10:20] = 0.9
    # start retroactive to frame 10; end = first silence frame after hangover
    assert segments(probs, CFG) == [(10, 20)]


def test_short_blip_rejected():
    probs = np.full(30, 0.1)
    probs[10:12] = 0.9  # 2 frames < min_speech_frames=3
    assert segments(probs, CFG) == []


def test_hysteresis_mid_prob_holds_speech_but_never_starts_it():
    # 0.5 is between offset (0.4) and onset (0.7)
    mid_only = np.full(30, 0.5)
    assert segments(mid_only, CFG) == []  # never crosses onset

    probs = np.full(40, 0.1)
    probs[5:10] = 0.9
    probs[10:20] = 0.5  # dips below onset but stays above offset
    probs[20:25] = 0.9
    assert segments(probs, CFG) == [(5, 25)]  # one segment, no flap


def test_hangover_bridges_short_gap_and_splits_long_gap():
    probs = np.full(60, 0.1)
    probs[5:15] = 0.9
    probs[18:28] = 0.9  # 3-frame gap < min_silence_frames=5 → bridged
    assert segments(probs, CFG) == [(5, 28)]

    probs = np.full(60, 0.1)
    probs[5:15] = 0.9
    probs[25:35] = 0.9  # 10-frame gap → split
    assert segments(probs, CFG) == [(5, 15), (25, 35)]


def test_open_segment_closed_at_stream_end():
    probs = np.full(20, 0.1)
    probs[10:] = 0.9
    assert segments(probs, CFG) == [(10, 20)]


def test_trailing_silence_shorter_than_hangover_still_ends_at_speech_edge():
    probs = np.full(23, 0.1)
    probs[10:20] = 0.9  # 3 trailing silence frames < hangover
    assert segments(probs, CFG) == [(10, 20)]


def test_streaming_push_matches_offline():
    rng = np.random.default_rng(0)
    probs = rng.random(500)
    pp = VadPostprocessor(CFG)
    events = [e for p in probs if (e := pp.push(float(p)))]
    events += pp.finish()
    starts = [i for kind, i in events if kind == "start"]
    ends = [i for kind, i in events if kind == "end"]
    assert list(zip(starts, ends, strict=True)) == segments(probs, CFG)


def test_config_from_sidecar():
    sidecar = {
        "recommended_threshold": 0.68,
        "postproc": {
            "offset_threshold": 0.45,
            "min_speech_frames": 3,
            "min_silence_frames": 25,
        },
    }
    cfg = PostprocConfig.from_sidecar(sidecar)
    assert cfg.onset_threshold == 0.68
    assert cfg.offset_threshold == 0.45
    assert cfg.min_speech_frames == 3
    assert cfg.min_silence_frames == 25
