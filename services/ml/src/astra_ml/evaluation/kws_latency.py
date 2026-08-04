"""CPU inference latency + real-time factor for the exported wake-word v2 graph.

Benches one 80 ms score step — window roll plus a full forward through log-mel and
BC-ResNet — at the cadence the Go detector runs. The frontend recomputes the whole
window every chunk by design (the graph is stateless), so that recompute is the cost.

RTF = compute_seconds / 0.08. RTF < 1 == faster than real time.

    uv run python -m astra_ml.evaluation.kws_latency --threads 1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from astra_ml.evaluation.kws_eval import DEFAULT_MODEL, INT16_SCALE, OnnxScorer
from astra_ml.training.kws_config import load_kws_config

SR = 16000
CHUNK = 1280
CHUNK_SECONDS = CHUNK / SR


def percentile(sorted_vals: list[float], pct: float) -> float:
    # ponytail: nearest-rank, no numpy needed for a benchmark
    idx = min(len(sorted_vals) - 1, int(pct / 100 * len(sorted_vals)))
    return sorted_vals[idx]


def bench_chunk(scorer: OnnxScorer, iters: int, warmup: int) -> dict:
    window = np.zeros(scorer.window_samples, dtype=np.float32)
    chunk = np.zeros(CHUNK, dtype=np.float32)

    def step() -> None:
        nonlocal window
        window = np.roll(window, -CHUNK)
        window[-CHUNK:] = chunk * INT16_SCALE
        scorer.score(window)

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
    parser.add_argument("--config", type=Path, default=Path("configs/ww_v2.yaml"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=1, help="1 == worst-case single core")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    cfg = load_kws_config(args.config)
    scorer = OnnxScorer(args.model, cfg.frontend.window_samples, threads=args.threads)
    report = {
        "device": "cpu",
        "threads": args.threads,
        "window_samples": cfg.frontend.window_samples,
        **bench_chunk(scorer, args.iters, args.warmup),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
