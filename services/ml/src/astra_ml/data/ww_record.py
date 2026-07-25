"""Mic recording helper for real "Astraa" clips — and for personal hard negatives.

Records 16 kHz mono s16le clips into session-stamped directories under
data.recordings_root and appends rows to its manifest.csv. Sessions are the
unit of the train/eval/test split (astra_ml.data.ww_recordings): record each
condition (room, distance, noise) as its own session and never mix speakers
within one.

Needs the `record` dependency group: uv sync --group record

    uv run python -m astra_ml.data.ww_record --session kitchen_quiet_0718 --n 25

--negative records into data.negative_recordings_root instead: the same speaker,
mic and room saying anything BUT the wake word. Pair it with --prompts to walk a
phrase list (one phrase per line) so the session actually covers the phonetic
neighbourhood instead of whatever comes to mind:

    uv run python -m astra_ml.data.ww_record --negative \\
        --session neg_quiet_near_0725 --prompts prompts/ww_hard_negatives.txt

Record negatives in the same rooms and at the same distances as the positive
sessions. They are the only thing in the whole negative pool that covers the
deployment channel, so a mismatch there is a mismatch the model never learns to
reject.
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


def read_prompts(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v1.yaml"))
    parser.add_argument("--session", required=True, help="e.g. kitchen_quiet_0718")
    parser.add_argument("--speaker", default="user")
    parser.add_argument("--env", default="", help="freeform condition note, e.g. 'music 2m'")
    parser.add_argument(
        "--n", type=int, default=None, help="clips to record (default: 25, or the prompt count)"
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument(
        "--negative",
        action="store_true",
        help="record hard negatives (anything but the wake word) into "
        "data.negative_recordings_root",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        help="file of phrases, one per line, walked in order (blank/# lines skipped). "
        "With --n unset the session length is the prompt count.",
    )
    args = parser.parse_args()

    cfg = load_ww_config(args.config)
    root = cfg.data.negative_recordings_root if args.negative else cfg.data.recordings_root
    session_dir = root / f"session_{args.session}"
    session_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(session_dir.glob("*.wav")))

    prompts = read_prompts(args.prompts) if args.prompts else []
    # An explicit --n still wins, so a long list can be recorded across several sittings
    # (each resumes at `existing`, which indexes into the same list).
    if args.n is not None:
        n = args.n
    elif prompts:
        n = len(prompts) - existing
    else:
        n = 25
    if n <= 0:
        print(f"nothing to record: {existing} clips already in {session_dir}")
        return

    rows = []
    for i in range(existing, existing + n):
        if prompts:
            say = prompts[i % len(prompts)]
        else:
            say = "something that is NOT the wake word" if args.negative else "the wake word"
        input(f"[{i - existing + 1}/{n}] press Enter, then say: {say}")
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
