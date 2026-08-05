"""Session-based split of real wake-word recordings.

Splits are assigned per session, never per clip: clips from one session share
room, mic, and day, so a per-clip split would leak train data into eval. Rules:

- sessions listed in data.frozen_test_sessions -> "test" (never trained on,
  never tuned against; the honesty backstop)
- remaining sessions -> deterministic hash: ~1 in 5 to "eval", rest "train"

Only "train" rows may enter training (pitch/speed-augmented there).

--negative applies the identical rules to data.negative_recordings_root (the
personal hard negatives). Their "train" rows become the negatives_user training
pool; the eval/test rows become the speech false-accept set ww_eval gates on, so
the number that measures "fires on any word I say" is measured on sessions
training never saw.
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


def assign_splits(rows: list[dict], frozen_test_sessions: list[str]) -> list[dict]:
    frozen = set(frozen_test_sessions)
    known = {r["session"] for r in rows}
    missing = frozen - known
    if missing:
        raise ValueError(f"frozen_test_sessions not present in manifest: {sorted(missing)}")

    out = []
    for r in rows:
        session = r["session"]
        split = "test" if session in frozen else _bucket(session)
        out.append({**r, "split": split})
    return out


def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def scan_negative_manifest(root: Path) -> list[dict]:
    """Rebuild manifest.csv rows from session_*/*.wav on disk, for negatives that landed
    in the folder without going through ww_record.py's --negative flow (e.g. copied in
    directly on a training machine)."""
    rows = []
    for session_dir in sorted(root.glob("session_*")):
        session = session_dir.name.removeprefix("session_")
        for wav in sorted(session_dir.glob("*.wav")):
            rows.append(
                {
                    "path": wav.relative_to(root).as_posix(),
                    "session": session,
                    "speaker": "user",
                    "env": "",
                }
            )
    return rows


def write_negative_manifest(root: Path) -> Path:
    rows = scan_negative_manifest(root)
    out = root / "manifest.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "session", "speaker", "env"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{out}: {len(rows)} clips from {root}/session_*")
    return out


def write_split_manifest(config_path: Path, negative: bool = False) -> Path:
    cfg = load_ww_config(config_path)
    if negative:
        root, frozen = cfg.data.negative_recordings_root, cfg.data.frozen_negative_test_sessions
        write_negative_manifest(root)
    else:
        root, frozen = cfg.data.recordings_root, cfg.data.frozen_test_sessions
    rows = assign_splits(load_manifest(root / "manifest.csv"), frozen)
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
    parser.add_argument(
        "--negative",
        action="store_true",
        help="split data.negative_recordings_root instead (hard negatives)",
    )
    args = parser.parse_args()
    write_split_manifest(args.config, negative=args.negative)


if __name__ == "__main__":
    main()
