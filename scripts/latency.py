#!/usr/bin/env python3
"""Turn a run's latency JSONL into the markdown table the README publishes.

The server writes one JSON line per completed chain (see internal/metrics),
interleaved with ordinary human log lines — so anything that is not JSON, or
not a latency line, is skipped rather than treated as an error.

    go run ./cmd/astra -verbose ... 2> runs/latency.jsonl
    python3 scripts/latency.py runs/latency.jsonl

stdlib only, on purpose: this reads a few hundred lines of JSON.
"""

import json
import math
import statistics
import sys

# Reported in chain order rather than sorted, so the table reads as a
# waterfall. user_speech is measured but deliberately not part of the headline:
# how long the speaker talked is not latency.
TURN_SPANS = ["wake_detect", "vad_detect", "user_speech",
              "endpoint_tail", "asr", "llm_ttft", "tts_ttfb"]
BARGE_SPANS = ["barge_detect", "cancel_send"]
EXCLUDED = {"user_speech", "wake_detect", "vad_detect"}


def percentile(values, q):
    """Nearest-rank percentile — no interpolation, so every number printed is
    one that was actually measured."""
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[k]


def load(path):
    turns, barges = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue  # a human log line
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ev") != "turn":
                continue
            (barges if rec.get("chain") == "barge" else turns).append(rec)
    return turns, barges


def table(title, records, span_names, total_key, total_label):
    if not records:
        print(f"\n_{title}: no chains recorded._")
        return
    print(f"\n**{title}** — {len(records)} chains\n")
    print("| span | p50 (ms) | p90 (ms) | n |")
    print("| --- | ---: | ---: | ---: |")
    for name in span_names:
        vals = [r["spans"][name] for r in records if name in r.get("spans", {})]
        if not vals:
            continue
        note = " *(not latency)*" if name == "user_speech" else ""
        print(f"| `{name}`{note} | {percentile(vals, .5)} | {percentile(vals, .9)} | {len(vals)} |")
    totals = [total_key(r) for r in records]
    totals = [t for t in totals if t is not None]
    if totals:
        print(f"| **{total_label}** | **{percentile(totals, .5)}** | "
              f"**{percentile(totals, .9)}** | {len(totals)} |")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <latency.jsonl>")
    turns, barges = load(sys.argv[1])

    table("Turn chain — wake word to first reply audio written", turns, TURN_SPANS,
          lambda r: r.get("headline_ms"), "time to first audio")
    table("Barge-in chain — speech onset to Cancel written", barges, BARGE_SPANS,
          lambda r: sum(v for k, v in r.get("spans", {}).items() if k not in EXCLUDED),
          "onset to cancel")

    if turns:
        # Naming the dominant span is the whole point of measuring before
        # optimising, so the script says it rather than leaving it to the eye.
        totals = {}
        for r in turns:
            for name, val in r.get("spans", {}).items():
                if name not in EXCLUDED:
                    totals.setdefault(name, []).append(val)
        worst = max(totals, key=lambda n: statistics.median(totals[n]))
        print(f"\nDominant span: `{worst}` "
              f"(p50 {percentile(totals[worst], .5)} ms). Optimise that one first.")
    print("\nEvery boundary is stamped server-side, so both chains end at a socket "
          "write; client-side playback latency is not included.")


if __name__ == "__main__":
    main()
