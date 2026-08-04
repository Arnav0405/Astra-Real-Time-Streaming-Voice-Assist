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
  models/      model architectures (vad.py: CNN + GRU + streaming wrapper;
               bcresnet.py + subspectralnorm.py: wake word v2)
  training/    plain-PyTorch training loops and YAML configs
  evaluation/  gate metrics, CHiME real-domain eval, Silero baseline
  export/      ONNX export (+ sidecar JSON) into ../../assets/models/
configs/     experiment configs (vad_v1.yaml, ww_v1.yaml, ww_v2.yaml)
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
# 1. frontends + piper voice; --acav also fetches the ~16 GB ACAV negatives (GPU machine only)
uv run python -m astra_ml.data.oww_assets --inspect --acav

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

## Wake word v2 — BC-ResNet from scratch

Config: `configs/ww_v2.yaml`. Replaces v1's two frozen OpenWakeWord graphs with a
trained log-mel + BC-ResNet-3 stack ("Broadcasted Residual Learning for Efficient
Keyword Spotting", Kim et al. 2021), and — the actual point — swaps v1's ambient-only
negative pool for 100 h of LibriSpeech connected speech. v1 measured 200 fa/hr and
345 speech fa/hr against gates of 20 and 5 because nothing in its negatives ever asked
it to reject a spoken word.

Both models ship side by side (`-ww-model` / `-ww-config` select at runtime) until v2
clears all five gates; only then does v1 and the `openwakeword` dependency come out.

Data *generation* is unchanged — v2 reuses the clips `ww_generate` / `ww_record`
already produced, plus LibriSpeech from the VAD pipeline's `data.generate`. Differences
from v1 worth knowing: there is no precompute step (windows are built and augmented in
DataLoader workers, and the mel runs on the GPU with the batch), and eval scores through
the exported ONNX rather than the torch checkpoint.

```sh
# 0. prerequisites: steps 1-4 of the v1 pipeline for the TTS/recorded clips, and
#    `data.generate` for LibriSpeech + RIRS_NOISES under datasets/source/

# 1. train (SGD, warmup -> lr 0.1 -> cosine, 75k steps)
uv run python -m astra_ml.training.train_kws --config configs/ww_v2.yaml   # tensorboard --logdir runs/ww_v2

# 2. export the single graph (log-mel + BC-ResNet + sigmoid) with a torch-vs-ORT parity gate
uv run python -m astra_ml.export.export_kws --checkpoint runs/ww_v2/best.pt

# 3. evaluate and tune trigger knobs in one pass over the corpora, then re-export
uv run python -m astra_ml.evaluation.kws_eval --update-sidecar
uv run python -m astra_ml.export.export_kws --checkpoint runs/ww_v2/best.pt

# 4. real-time factor on one core, and the Go parity fixtures
uv run python -m astra_ml.evaluation.kws_latency --threads 1
uv run python -m astra_ml.export.golden_ww --name ww_v2
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned (ML dependency wheel compatibility).

```sh
uv sync          # create .venv and install dev tools
uv run pytest    # run tests
uv run ruff check .
```

Or from the repo root: `make format`, `make lint`, `make test`.
