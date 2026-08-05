"""Tests for the session-based recording split discipline."""

import pytest

from astra_ml.data.ww_recordings import assign_splits, scan_negative_manifest


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


def test_frozen_sessions_become_test():
    rows = rows_for({"a": 3, "b": 2})
    got = assign_splits(rows, frozen_test_sessions=["a"])
    by_session = {r["session"]: r["split"] for r in got}
    assert by_session["a"] == "test"
    assert by_session["b"] in ("train", "eval")


def test_split_is_per_session_and_deterministic():
    rows = rows_for({f"s{i}": 4 for i in range(20)})
    got1 = assign_splits(rows, [])
    got2 = assign_splits(rows, [])
    assert got1 == got2
    for session in {r["session"] for r in got1}:
        splits = {r["split"] for r in got1 if r["session"] == session}
        assert len(splits) == 1  # never split within a session


def test_hash_bucket_produces_both_splits():
    rows = rows_for({f"s{i}": 1 for i in range(50)})
    splits = {r["split"] for r in assign_splits(rows, [])}
    assert splits == {"train", "eval"}


def test_unknown_session_rejected():
    rows = rows_for({"a": 1})
    with pytest.raises(ValueError, match="frozen_test_sessions"):
        assign_splits(rows, ["nope"])


def test_scan_negative_manifest_reads_wavs_from_session_dirs(tmp_path):
    for session, n in {"neg_a": 2, "neg_b": 1}.items():
        d = tmp_path / f"session_{session}"
        d.mkdir()
        for i in range(n):
            (d / f"{i:03d}.wav").write_bytes(b"")
    (tmp_path / "not_a_session").mkdir()  # ignored: no session_ prefix

    rows = scan_negative_manifest(tmp_path)

    assert {r["session"] for r in rows} == {"neg_a", "neg_b"}
    assert sorted(r["path"] for r in rows) == [
        "session_neg_a/000.wav",
        "session_neg_a/001.wav",
        "session_neg_b/000.wav",
    ]
