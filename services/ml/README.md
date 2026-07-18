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

## Wake word pipeline (Phase 4)

Config: `configs/ww_v1.yaml`. Steps 1, 2 and recording run anywhere; generation
at scale, precompute, and training belong on the GPU machine.

```sh
# 1. frontends + piper voice (also prints the ~16 GB ACAV negatives curl command)
uv run python -m astra_ml.data.oww_assets --inspect

# 2. pronunciation smoke test — listen to a few clips per spelling, adjust data.spellings
uv run python -m astra_ml.data.ww_generate --smoke 4   # datasets/ww_tts/smoke/<spelling>/

# 3. record real clips (needs: uv sync --group record); one session per room/noise condition,
#    ~150+ of you across sessions plus a couple family sessions
uv run python -m astra_ml.data.ww_record --session kitchen_quiet_0718 --n 25
#    then freeze ~2 sessions as data.frozen_test_sessions (never trained/tuned against),
#    list family sessions in data.family_sessions, and build the split:
uv run python -m astra_ml.data.ww_recordings

# 4. full TTS generation (30k positives + adversarials), features, training
uv run python -m astra_ml.data.ww_generate
uv run python -m astra_ml.training.train_ww precompute
uv run python -m astra_ml.training.train_ww train        # tensorboard --logdir runs/ww

# 5. evaluate against the gate metrics, tune trigger knobs, export
uv run python -m astra_ml.evaluation.ww_eval --checkpoint runs/ww/best.pt
uv run python -m astra_ml.evaluation.tune_ww --checkpoint runs/ww/best.pt
uv run python -m astra_ml.export.export_ww --checkpoint runs/ww/best.pt   # assets/models/wakeword/

# 6. regenerate the Go parity fixtures (now that ww_v1 artifacts exist)
uv run python -m astra_ml.export.golden_ww
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned (ML dependency wheel compatibility).

```sh
uv sync          # create .venv and install dev tools
uv run pytest    # run tests
uv run ruff check .
```

Or from the repo root: `make format`, `make lint`, `make test`.
