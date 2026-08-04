"""CPU inference latency + real-time factor for the wake-word pipeline.

Benches one 80 ms score step (window roll + frontend features + head forward)
at the exact cadence ww_eval.stream_scores uses in production — the frontend
recomputes the full window every chunk, so that recompute is the real cost,
not just the (tiny) head forward.

RTF = compute_seconds / chunk_seconds. RTF < 1 == faster than real time.

Usage:
    uv run python -m astra_ml.evaluation.ww_latency --config configs/ww_v1.yaml --checkpoint runs/ww/best.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from astra_ml.audio.ww_frontend import WwFrontend
from astra_ml.training.train_ww import INT16_SCALE, load_head
from astra_ml.training.ww_config import load_ww_config

SR = 16000
CHUNK = 1280  # 80 ms score step, matches ww_eval.stream_scores cadence
CHUNK_SECONDS = CHUNK / SR


def percentile(sorted_vals: list[float], pct: float) -> float:
    # ponytail: nearest-rank, no numpy needed for a benchmark
    idx = min(len(sorted_vals) - 1, int(pct / 100 * len(sorted_vals)))
    return sorted_vals[idx]


@torch.no_grad()
def bench_chunk(frontend: WwFrontend, head, iters: int, warmup: int) -> dict:
    window = np.zeros(frontend.cfg.window_samples, dtype=np.float32)
    chunk = np.zeros(CHUNK, dtype=np.float32)

    def step() -> None:
        nonlocal window
        window = np.roll(window, -CHUNK)
        window[-CHUNK:] = chunk * INT16_SCALE
        feats = frontend.features(window)
        head(torch.from_numpy(feats.astype(np.float32))[None])

    for _ in range(warmup):
        step()

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        latencies.append(time.perf_counter() - t0)
    latencies.sort()

    mean = sum(latencies) / len(latencies)
    return {
        "chunk_latency_ms_mean": mean * 1e3,
        "chunk_latency_ms_p50": percentile(latencies, 50) * 1e3,
        "chunk_latency_ms_p95": percentile(latencies, 95) * 1e3,
        "chunk_latency_ms_max": latencies[-1] * 1e3,
        "rtf_chunk": mean / CHUNK_SECONDS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v1.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)  # 1 == worst-case single core, matches edge deploy
    cfg = load_ww_config(args.config)
    frontend = WwFrontend.from_onnx(cfg.data.frontends_dir)
    head = load_head(args.checkpoint)

    report = {
        "device": "cpu",
        "threads": args.threads,
        **bench_chunk(frontend, head, args.iters, args.warmup),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
