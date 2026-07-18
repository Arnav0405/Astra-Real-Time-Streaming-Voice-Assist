"""Download the frozen OpenWakeWord frontends and piper TTS voices (gitignored).

`--inspect` probes the downloaded frontends with onnxruntime and pins the
mel/embedding arithmetic the ww_v1.json sidecar and the Go runtime depend on.
The ACAV negative-features file is ~16 GB, so this script only prints the
download command for it (a one-time GPU-machine action).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from openwakeword import FEATURE_MODELS
from openwakeword.utils import download_file

from astra_ml.training.ww_config import load_ww_config

ACAV_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features"
    "/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
)
PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def frontend_urls() -> list[str]:
    return [m["download_url"].replace(".tflite", ".onnx") for m in FEATURE_MODELS.values()]


def voice_urls(name: str) -> list[str]:
    lang_region, voice, quality = name.split("-")
    lang = lang_region.split("_")[0]
    base = f"{PIPER_VOICES_BASE}/{lang}/{lang_region}/{voice}/{quality}/{name}.onnx"
    return [base, f"{base}.json"]


def download(urls: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for url in urls:
        target = dest / url.split("/")[-1]
        if target.exists():
            print(f"exists: {target}")
        else:
            download_file(url, str(dest))


def inspect_frontends(frontends_dir: Path) -> dict:
    """Empirically pin the feature arithmetic. Returns the numbers the sidecar carries."""
    melspec = ort.InferenceSession(str(frontends_dir / "melspectrogram.onnx"))
    embedding = ort.InferenceSession(str(frontends_dir / "embedding_model.onnx"))

    mel_in = melspec.get_inputs()[0].name
    frames = {}
    for n in (1280, 1600, 1760, 2560, 5120):
        out = melspec.run(None, {mel_in: np.zeros((1, n), dtype=np.float32)})[0]
        frames[n] = out.shape[-2]
        mel_bins = out.shape[-1]

    # frames(n) is affine in n: frames = (n - window) // hop + 1
    hop = (2560 - 1280) // (frames[2560] - frames[1280])
    # infer effective window: n - (frames-1)*hop
    window = 1280 - (frames[1280] - 1) * hop

    emb_in = embedding.get_inputs()[0]
    emb_out = embedding.run(None, {emb_in.name: np.zeros([1, 76, 32, 1], dtype=np.float32)})[0]
    emb_window = 76
    emb_stride = 8
    emb_dim = int(np.prod(emb_out.shape[1:]))
    head_frames = 16
    window_frames_total = emb_window + (head_frames - 1) * emb_stride  # 196
    window_samples = (window_frames_total - 1) * hop + window

    info = {
        "mel": {
            "bins": int(mel_bins),
            "hop_samples": int(hop),
            "window_samples": int(window),
            "frames_by_input": {str(k): int(v) for k, v in frames.items()},
            "transform": {"scale": 0.1, "offset": 2.0},  # applied outside the graph (OWW utils)
            "input_name": mel_in,
            "output_shape": [int(x) for x in np.array(out.shape)],
        },
        "embedding": {
            "window_frames": emb_window,
            "stride_frames": emb_stride,
            "dim": emb_dim,
            "input_name": emb_in.name,
            "output_shape": [int(x) for x in emb_out.shape],
        },
        "head": {"input_frames": head_frames},
        "window_frames_total": window_frames_total,
        "window_samples": int(window_samples),
    }
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v1.yaml"))
    parser.add_argument("--inspect", action="store_true", help="probe frontend arithmetic")
    args = parser.parse_args()
    cfg = load_ww_config(args.config)

    download(frontend_urls(), cfg.data.frontends_dir)
    for name in cfg.data.voices:
        download(voice_urls(name), cfg.data.voices_dir)

    if not cfg.data.acav_features.exists():
        cfg.data.acav_features.parent.mkdir(parents=True, exist_ok=True)
        print("\nACAV negative features (~16 GB) not present. On the GPU machine run:")
        print(f"  curl -L -o {cfg.data.acav_features} '{ACAV_URL}'")

    if args.inspect:
        info = inspect_frontends(cfg.data.frontends_dir)
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
