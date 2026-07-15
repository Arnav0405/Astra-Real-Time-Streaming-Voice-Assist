"""CPU inference latency + real-time factor for the streaming VAD.

Streaming (stream_step, one 20 ms frame + state) is the deployment path, so it's
the headline number. Batch forward RTF included for offline/eval throughput.

RTF = compute_seconds / audio_seconds. RTF < 1 == faster than real time.

Usage:
    uv run python -m astra_ml.evaluation.latency --checkpoint runs/vad/best.pt
"""

import argparse
import json
import time
from pathlib import Path

import torch

from astra_ml.audio.dft_mel import FRAME_SAMPLES, SAMPLE_RATE
from astra_ml.models.vad import VadModel, zero_state

FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE  # 0.02


def percentile(sorted_vals: list[float], pct: float) -> float:
    # ponytail: nearest-rank, no numpy needed for a benchmark
    idx = min(len(sorted_vals) - 1, int(pct / 100 * len(sorted_vals)))
    return sorted_vals[idx]


@torch.no_grad()
def bench_stream(model: VadModel, iters: int, warmup: int) -> dict:
    frame = torch.zeros(1, FRAME_SAMPLES)
    state = zero_state(1)
    for _ in range(warmup):
        _, state = model.stream_step(frame, state)

    latencies = []
    state = zero_state(1)
    for _ in range(iters):
        t0 = time.perf_counter()
        _, state = model.stream_step(frame, state)
        latencies.append(time.perf_counter() - t0)
    latencies.sort()

    mean = sum(latencies) / len(latencies)
    return {
        "frame_latency_ms_mean": mean * 1e3,
        "frame_latency_ms_p50": percentile(latencies, 50) * 1e3,
        "frame_latency_ms_p95": percentile(latencies, 95) * 1e3,
        "frame_latency_ms_max": latencies[-1] * 1e3,
        "rtf_stream": mean / FRAME_SECONDS,
    }


@torch.no_grad()
def bench_batch(model: VadModel, seconds: float, iters: int, warmup: int) -> dict:
    n_frames = int(seconds * SAMPLE_RATE) // FRAME_SAMPLES
    pcm = torch.zeros(1, n_frames * FRAME_SAMPLES)
    for _ in range(warmup):
        model(pcm)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(pcm)
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    audio_seconds = n_frames * FRAME_SECONDS
    return {
        "batch_clip_seconds": audio_seconds,
        "batch_compute_ms_mean": mean * 1e3,
        "rtf_batch": mean / audio_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)  # 1 == worst-case single core, matches edge deploy
    model = VadModel()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    report = {
        "device": "cpu",
        "threads": args.threads,
        **bench_stream(model, args.iters, args.warmup),
        **bench_batch(model, seconds=4.0, iters=max(args.iters // 40, 10), warmup=10),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
