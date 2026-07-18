"""Synthetic wake-word data: TTS positives and adversarial negatives.

Uses piper-sample-generator's ONNX voice path (`generate_samples_onnx`) with a
multi-speaker piper voice (en_US-libritts_r-medium: 904 speakers) crossed with
length/noise scale grids. The upstream `.pt` generator path needs the
unpublished `piper_train` package, whose top-level import breaks the module, so
a stub is installed before importing (ONNX path never touches it).

Piper voices synthesize at their native rate (22050 Hz for libritts_r); clips
are resampled in place to 16 kHz mono s16le. `--smoke` generates a handful of
clips per spelling for pronunciation checks by ear.
"""

import argparse
import csv
import sys
import types
from pathlib import Path

import soundfile as sf
import soxr

from astra_ml.training.ww_config import WwConfig, load_ww_config


def _stub_piper_train() -> None:
    # ponytail: fake module tree; delete when psg guards its .pt-path import
    if "piper_train" in sys.modules:
        return
    pt = types.ModuleType("piper_train")
    vits = types.ModuleType("piper_train.vits")
    vits.commons = types.ModuleType("piper_train.vits.commons")
    pt.vits = vits
    sys.modules["piper_train"] = pt
    sys.modules["piper_train.vits"] = vits
    sys.modules["piper_train.vits.commons"] = vits.commons


_stub_piper_train()

from piper_sample_generator.__main__ import generate_samples_onnx  # noqa: E402

TARGET_SR = 16000
LENGTH_SCALES = (0.7, 0.85, 1.0, 1.15, 1.3)
NOISE_SCALES = (0.333, 0.667)
NOISE_SCALE_WS = (0.6, 0.8)


def generate_set(texts: list[str], n: int, out_dir: Path, voice_paths: list[Path]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_samples_onnx(
        text=texts,
        output_dir=str(out_dir),
        model=[str(p) for p in voice_paths],
        max_samples=n,
        length_scales=LENGTH_SCALES,
        noise_scales=NOISE_SCALES,
        noise_scale_ws=NOISE_SCALE_WS,
    )
    wavs = sorted(out_dir.glob("*.wav"), key=lambda p: int(p.stem))
    for wav in wavs:
        resample_to_16k(wav)
    return wavs


def resample_to_16k(path: Path) -> None:
    audio, sr = sf.read(path, dtype="float32")
    if sr == TARGET_SR:
        return
    resampled = soxr.resample(audio, sr, TARGET_SR)
    sf.write(path, resampled, TARGET_SR, subtype="PCM_16")


def slug(phrase: str) -> str:
    return phrase.replace(" ", "_")


def generate_all(cfg: WwConfig) -> Path:
    """Full generation run. Returns the manifest path."""
    data = cfg.data
    voices = data.voice_paths()
    rows = []

    n_total = data.n_positives + data.n_positives_val
    pos_dir = data.tts_out / "positives"
    wavs = generate_set(list(data.spellings), n_total, pos_dir, voices)
    for i, wav in enumerate(wavs):
        split = "val" if i >= data.n_positives else "train"
        rows.append((wav.relative_to(data.tts_out).as_posix(), "positive", "astraa", split))

    for phrase in data.adversarial_phrases:
        adv_dir = data.tts_out / "adversarial" / slug(phrase)
        wavs = generate_set([phrase], data.n_adversarial_per_phrase, adv_dir, voices)
        for wav in wavs:
            rows.append((wav.relative_to(data.tts_out).as_posix(), "adversarial", phrase, "train"))

    manifest = data.tts_out / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "phrase", "split"])
        writer.writerows(rows)
    return manifest


def generate_smoke(cfg: WwConfig, n_per_spelling: int) -> None:
    voices = cfg.data.voice_paths()
    for spelling in cfg.data.spellings:
        out = cfg.data.tts_out / "smoke" / spelling
        generate_set([spelling], n_per_spelling, out, voices)
        print(f"{spelling}: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v1.yaml"))
    parser.add_argument(
        "--smoke", type=int, default=0, metavar="N", help="generate N clips per spelling and stop"
    )
    args = parser.parse_args()
    cfg = load_ww_config(args.config)
    if args.smoke:
        generate_smoke(cfg, args.smoke)
    else:
        manifest = generate_all(cfg)
        print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
