"""Tests for the wake-word trigger state machine."""

from astra_ml.postproc_ww import WwPostprocConfig, WwPostprocessor

CFG = WwPostprocConfig(threshold=0.5, patience_frames=2, refractory_frames=100)


def run_seq(pp, scores, start_frame=0, step=4):
    """Push scores at 80 ms cadence (4 transport frames per step); collect triggers."""
    out = []
    frame = start_frame
    for s in scores:
        frame += step
        t = pp.push(s, frame - 1)  # frame index of the chunk's last 20 ms frame
        if t is not None:
            out.append(t)
    return out


def test_single_step_blip_below_patience_does_not_fire():
    pp = WwPostprocessor(CFG)
    assert run_seq(pp, [0.9, 0.1, 0.9, 0.1, 0.9]) == []


def test_fires_after_patience_consecutive_steps():
    pp = WwPostprocessor(CFG)
    # steps at frames 3, 7, 11, ... ; second qualifying step (frame 7) triggers
    assert run_seq(pp, [0.9, 0.9, 0.1]) == [7]


def test_threshold_is_inclusive():
    pp = WwPostprocessor(CFG)
    assert run_seq(pp, [0.5, 0.5]) == [7]


def test_refractory_suppresses_then_allows():
    pp = WwPostprocessor(CFG)
    # 30 consecutive high steps: trigger at frame 7, refractory 100 frames
    # suppresses until frame >= 107 -> next trigger at frame 107.
    triggers = run_seq(pp, [0.9] * 30)
    assert triggers == [7, 107]


def test_gate_reset_clears_patience_run():
    pp = WwPostprocessor(CFG)
    assert pp.push(0.9, 3) is None
    pp.gate_reset()
    assert pp.push(0.9, 103) is None  # run restarted, needs another consecutive step
    assert pp.push(0.9, 107) == 107


def test_refractory_survives_gate_reset():
    pp = WwPostprocessor(CFG)
    assert run_seq(pp, [0.9, 0.9]) == [7]
    pp.gate_reset()
    # new gate shortly after: still inside refractory window
    assert pp.push(0.9, 20) is None
    assert pp.push(0.9, 24) is None  # run == patience but frame 24 - 7 < 100
    # much later gate: beyond refractory
    pp.gate_reset()
    assert pp.push(0.9, 200) is None
    assert pp.push(0.9, 204) == 204


def test_from_sidecar():
    sidecar = {
        "recommended_threshold": 0.62,
        "postproc": {"patience_frames": 3, "refractory_frames": 150},
    }
    cfg = WwPostprocConfig.from_sidecar(sidecar)
    assert cfg == WwPostprocConfig(0.62, 3, 150)
