"""Real-domain gate: FA rate on CHiME non-speech chunks (< 5 %), plus weak speech recall.

Usage:
    uv run python -m astra_ml.evaluation.chime_eval --config configs/vad_v1.yaml \
        --checkpoint runs/vad/best.pt --threshold 0.5
"""

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf
import torch

from astra_ml.audio.dft_mel import FRAME_SAMPLES
from astra_ml.models.vad import VadModel
from astra_ml.training.config import load_config
from astra_ml.training.train import pick_device


def read_manifest(path: Path) -> list[Path]:
    with open(path) as f:
        return [Path(row[1]) for row in csv.reader(f)]


@torch.no_grad()
def chunk_probs(model: VadModel, wav: Path, device) -> torch.Tensor:
    pcm, _ = sf.read(wav, dtype="float32", always_2d=True)
    pcm = pcm[: len(pcm) // FRAME_SAMPLES * FRAME_SAMPLES, 0]
    return model(torch.from_numpy(pcm).unsqueeze(0).to(device)).cpu().flatten()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vad_v1.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    model = VadModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    fa_frames = total_frames = 0
    for wav in read_manifest(cfg.data.chime_prepared / "eval_nonspeech.csv"):
        probs = chunk_probs(model, wav, device)
        fa_frames += int((probs >= args.threshold).sum())
        total_frames += probs.numel()
    fa_rate = fa_frames / total_frames

    detected = chunks = 0
    for wav in read_manifest(cfg.data.chime_prepared / "eval_speech.csv"):
        detected += int(chunk_probs(model, wav, device).max() >= args.threshold)
        chunks += 1

    report = {
        "fa_rate_nonspeech": fa_rate,
        "gate_fa_below_5pct": fa_rate < 0.05,
        "speech_chunk_detection": detected / max(chunks, 1),
        "threshold": args.threshold,
    }
    out = cfg.training.runs_dir / "chime_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
