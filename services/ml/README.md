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
src/astra_ml/
  audio/       audio processing (DFT-matmul log-mel frontend)
  data/        dataset code: LibriParty labels/Dataset, CHiME manifests, generation wrapper
  models/      model architectures (vad.py: CNN + GRU + streaming wrapper)
  training/    plain-PyTorch training loop and YAML config
  evaluation/  gate metrics, CHiME real-domain eval, Silero baseline
  export/      ONNX export (+ sidecar JSON) into ../../assets/models/
configs/     experiment configs (vad_v1.yaml)
datasets/    local dataset storage (gitignored)
tests/       dataset-free test suite (runs anywhere)
```

## VAD pipeline (GPU machine)

```sh
uv run python -m astra_ml.data.chime --chime-root datasets/chime_home --out datasets/chime_prepared
uv run python -m astra_ml.data.generate --config configs/vad_v1.yaml   # downloads + generates LibriParty
uv run python -m astra_ml.training.train --config configs/vad_v1.yaml  # tensorboard --logdir runs/vad
uv run python -m astra_ml.evaluation.eval --config configs/vad_v1.yaml --checkpoint runs/vad/best.pt
uv run python -m astra_ml.evaluation.chime_eval --config configs/vad_v1.yaml --checkpoint runs/vad/best.pt --threshold <from eval_report.json>
uv run python -m astra_ml.export.export --checkpoint runs/vad/best.pt --threshold <same>
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned (ML dependency wheel compatibility).

```sh
uv sync          # create .venv and install dev tools
uv run pytest    # run tests
uv run ruff check .
```

Or from the repo root: `make format`, `make lint`, `make test`.
