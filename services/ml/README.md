# Astra ML

Python model development — the offline path of Astra.

## Responsibilities

- Train the custom VAD model (PyTorch)
- Train the custom wake word model (OpenWakeWord)
- Evaluate models against held-out data
- Export models to ONNX into `../../assets/models/` for the Go runtime

This service never serves traffic. See [docs/architecture.md](../../docs/architecture.md) for boundaries.

## Layout

```
audio/       audio processing utilities (feature extraction, resampling)
datasets/    local dataset storage (gitignored) and dataset definitions
models/      model architectures
  vad/         voice activity detection
  wakeword/    wake word detection
training/    training loops and experiment configs
evaluation/  metrics and evaluation harnesses
export/      ONNX export and validation
tests/       test suite
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned (ML dependency wheel compatibility).

```sh
uv sync          # create .venv and install dev tools
uv run pytest    # run tests
uv run ruff check .
```

Or from the repo root: `make format`, `make lint`, `make test`.
