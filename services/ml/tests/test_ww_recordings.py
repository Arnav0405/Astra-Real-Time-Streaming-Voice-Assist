"""Tests for the session-based recording split discipline."""

import pytest

from astra_ml.data.ww_recordings import assign_splits


def rows_for(sessions: dict[str, int]) -> list[dict]:
    out = []
    for session, n in sessions.items():
        for i in range(n):
            out.append(
                {
                    "path": f"session_{session}/{i:03d}.wav",
                    "session": session,
                    "speaker": "user",
                    "env": "",
                }
            )
    return out


def test_frozen_sessions_become_test_family_eval_only():
    rows = rows_for({"a": 3, "b": 2, "mom": 2})
    got = assign_splits(rows, frozen_test_sessions=["a"], family_sessions=["mom"])
    by_session = {r["session"]: r["split"] for r in got}
    assert by_session["a"] == "test"
    assert by_session["mom"] == "eval_only"
    assert by_session["b"] in ("train", "eval")


def test_split_is_per_session_and_deterministic():
    rows = rows_for({f"s{i}": 4 for i in range(20)})
    got1 = assign_splits(rows, [], [])
    got2 = assign_splits(rows, [], [])
    assert got1 == got2
    for session in {r["session"] for r in got1}:
        splits = {r["split"] for r in got1 if r["session"] == session}
        assert len(splits) == 1  # never split within a session


def test_hash_bucket_produces_both_splits():
    rows = rows_for({f"s{i}": 1 for i in range(50)})
    splits = {r["split"] for r in assign_splits(rows, [], [])}
    assert splits == {"train", "eval"}


def test_overlapping_lists_rejected():
    rows = rows_for({"a": 1})
    with pytest.raises(ValueError, match="both"):
        assign_splits(rows, ["a"], ["a"])


def test_unknown_session_rejected():
    rows = rows_for({"a": 1})
    with pytest.raises(ValueError, match="frozen_test_sessions"):
        assign_splits(rows, ["nope"], [])
    with pytest.raises(ValueError, match="family_sessions"):
        assign_splits(rows, [], ["nope"])
