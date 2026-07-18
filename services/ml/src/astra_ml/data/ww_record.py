"""Mic recording helper for real "Astraa" clips.

Records 16 kHz mono s16le clips into session-stamped directories under
data.recordings_root and appends rows to its manifest.csv. Sessions are the
unit of the train/eval/test split (astra_ml.data.ww_recordings): record each
condition (room, distance, noise) as its own session and never mix speakers
within one.

Needs the `record` dependency group: uv sync --group record

    uv run python -m astra_ml.data.ww_record --session kitchen_quiet_0718 --n 25
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from astra_ml.training.ww_config import load_ww_config

SR = 16000


def record_clip(seconds: float) -> np.ndarray:
    import sounddevice as sd

    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16")
    sd.wait()
    return audio[:, 0]


def append_manifest(manifest: Path, rows: list[tuple[str, str, str, str]]) -> None:
    new = not manifest.exists()
    with open(manifest, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["path", "session", "speaker", "env"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v1.yaml"))
    parser.add_argument("--session", required=True, help="e.g. kitchen_quiet_0718")
    parser.add_argument("--speaker", default="user")
    parser.add_argument("--env", default="", help="freeform condition note, e.g. 'music 2m'")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    cfg = load_ww_config(args.config)
    root = cfg.data.recordings_root
    session_dir = root / f"session_{args.session}"
    session_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(session_dir.glob("*.wav")))

    rows = []
    for i in range(existing, existing + args.n):
        input(f"[{i - existing + 1}/{args.n}] press Enter, then say the wake word...")
        audio = record_clip(args.seconds)
        path = session_dir / f"{i:03d}.wav"
        sf.write(path, audio, SR, subtype="PCM_16")
        peak = np.abs(audio).max() / 32768
        print(f"  saved {path.name} (peak {peak:.2f})" + ("  ⚠ very quiet" if peak < 0.05 else ""))
        rows.append((path.relative_to(root).as_posix(), args.session, args.speaker, args.env))

    append_manifest(root / "manifest.csv", rows)
    print(f"{len(rows)} clips appended to {root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
