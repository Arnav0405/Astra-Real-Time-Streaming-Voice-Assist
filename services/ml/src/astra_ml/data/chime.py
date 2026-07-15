"""CHiME-Home: manifests + background-noise wavs for LibriParty generation.

Chunk annotations are 4 s, multi-label (c/m/f = speech, v = TV → contains speech,
p/b/o = non-speech). Only strong-agreement ("refined") chunk lists are used, so
background/non-speech chunks are reliably speech-free:
  - development_chunks_refined → training backgrounds
  - evaluation_chunks_refined  → real-domain eval (FA rate on non-speech, weak speech eval)

Usage:
    uv run python -m astra_ml.data.chime --chime-root datasets/chime_home \
        --out datasets/chime_prepared
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from astra_ml.audio.dft_mel import SAMPLE_RATE

SPEECH_LABELS = set("cmfv")
BACKGROUND_GROUP = 15  # 15 × 4 s chunks → 60 s background wavs


def read_majority_vote(chunk_csv: Path) -> str:
    with open(chunk_csv) as f:
        row = dict(csv.reader(f))
    return row.get("majorityvote", "")


def load_chunk_list(list_csv: Path) -> list[str]:
    with open(list_csv) as f:
        return [row[1] for row in csv.reader(f) if len(row) == 2]


def classify(chime_root: Path, chunk_names: list[str]) -> tuple[list[str], list[str]]:
    """→ (speech chunks, non-speech chunks); chunks with empty majority vote dropped."""
    speech, nonspeech = [], []
    for name in chunk_names:
        vote = read_majority_vote(chime_root / "chunks" / f"{name}.csv")
        if not vote:
            continue
        (speech if set(vote) & SPEECH_LABELS else nonspeech).append(name)
    return speech, nonspeech


def wav_path(chime_root: Path, chunk_name: str) -> Path:
    return chime_root / "chunks" / f"{chunk_name}.16kHz.wav"


def write_manifest(path: Path, chime_root: Path, chunks: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for name in chunks:
            writer.writerow([name, wav_path(chime_root, name)])


def build_backgrounds(chime_root: Path, out_dir: Path, chunks: list[str]) -> int:
    """Concatenate non-speech chunks into 60 s wavs (LibriParty backgrounds_root)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for i in range(0, len(chunks) - BACKGROUND_GROUP + 1, BACKGROUND_GROUP):
        parts = [
            sf.read(wav_path(chime_root, name), dtype="float32", always_2d=True)[0][:, 0]
            for name in chunks[i : i + BACKGROUND_GROUP]
        ]
        sf.write(out_dir / f"chime_bg_{n_files:03d}.wav", np.concatenate(parts), SAMPLE_RATE)
        n_files += 1
    return n_files


def prepare(chime_root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dev = load_chunk_list(chime_root / "development_chunks_refined.csv")
    ev = load_chunk_list(chime_root / "evaluation_chunks_refined.csv")

    _, dev_nonspeech = classify(chime_root, dev)
    ev_speech, ev_nonspeech = classify(chime_root, ev)

    backgrounds_dir = out / "backgrounds"
    shutil.rmtree(backgrounds_dir, ignore_errors=True)
    n_bg = build_backgrounds(chime_root, backgrounds_dir, dev_nonspeech)

    write_manifest(out / "dev_nonspeech.csv", chime_root, dev_nonspeech)
    write_manifest(out / "eval_speech.csv", chime_root, ev_speech)
    write_manifest(out / "eval_nonspeech.csv", chime_root, ev_nonspeech)
    print(
        f"backgrounds: {n_bg} × 60 s from {len(dev_nonspeech)} dev non-speech chunks\n"
        f"dev: {len(dev_nonspeech)} non-speech chunks (augmentation manifest)\n"
        f"eval: {len(ev_speech)} speech / {len(ev_nonspeech)} non-speech chunks"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chime-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.chime_root, args.out)


if __name__ == "__main__":
    main()
