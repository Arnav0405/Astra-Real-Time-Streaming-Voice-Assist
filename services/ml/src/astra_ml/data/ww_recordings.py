"""Session-based split of real wake-word recordings.

Splits are assigned per session, never per clip: clips from one session share
room, mic, and day, so a per-clip split would leak train data into eval. Rules:

- sessions listed in data.frozen_test_sessions -> "test" (never trained on,
  never tuned against; the honesty backstop)
- sessions listed in data.family_sessions -> "eval_only" (reported, not gating,
  never trained on)
- remaining sessions -> deterministic hash: ~1 in 5 to "eval", rest "train"

Only "train" rows may enter training (pitch/speed-augmented there).
"""

import argparse
import csv
import hashlib
from pathlib import Path

from astra_ml.training.ww_config import load_ww_config

EVAL_ONE_IN = 5


def _bucket(session: str) -> str:
    digest = hashlib.sha256(session.encode()).digest()
    return "eval" if digest[0] % EVAL_ONE_IN == 0 else "train"


def assign_splits(
    rows: list[dict],
    frozen_test_sessions: list[str],
    family_sessions: list[str],
) -> list[dict]:
    frozen = set(frozen_test_sessions)
    family = set(family_sessions)
    if frozen & family:
        raise ValueError(f"sessions in both frozen_test and family lists: {frozen & family}")
    known = {r["session"] for r in rows}
    for name, listed in (("frozen_test_sessions", frozen), ("family_sessions", family)):
        missing = listed - known
        if missing:
            raise ValueError(f"{name} not present in manifest: {sorted(missing)}")

    out = []
    for r in rows:
        session = r["session"]
        if session in frozen:
            split = "test"
        elif session in family:
            split = "eval_only"
        else:
            split = _bucket(session)
        out.append({**r, "split": split})
    return out


def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def write_split_manifest(config_path: Path) -> Path:
    cfg = load_ww_config(config_path)
    root = cfg.data.recordings_root
    rows = assign_splits(
        load_manifest(root / "manifest.csv"),
        cfg.data.frozen_test_sessions,
        cfg.data.family_sessions,
    )
    out = root / "manifest_split.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"{out}: {counts}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v1.yaml"))
    args = parser.parse_args()
    write_split_manifest(args.config)


if __name__ == "__main__":
    main()
