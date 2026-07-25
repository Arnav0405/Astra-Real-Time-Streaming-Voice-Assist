"""Shared fixtures for wake-word tests: a fully-wired tmp_path WwConfig."""

import pytest

from astra_ml.training.ww_config import load_ww_config

WW_CONFIG_TEMPLATE = """\
data:
  frontends_dir: {root}/frontends
  voices_dir: {root}/voices
  voices: [fake-voice-medium]
  tts_out: {root}/tts
  spellings: [astraa, astra]
  n_positives: 6
  n_positives_val: 2
  adversarial_phrases: [astro, ad astra]
  n_adversarial_per_phrase: 3
  acav_features: {root}/acav.npy
  acav_subsample: 10
  negative_audio_dirs: []
  recordings_root: {root}/rec
  negative_recordings_root: {root}/negrec
augment:
  rir_dir: {root}/rir
  rir_prob: 0.5
  noise_prob: 0.6
  snr_db_range: [3.0, 25.0]
  noise_dirs: []
  user_pitch_semitones: [-2.0, 2.0]
  user_speed_range: [0.9, 1.1]
  augment_rounds_user: 2
  augment_rounds_user_negative: 2
training:
  batch_size: 8
  steps: 6
  lr: 1.0e-3
  layer_size: 8
  max_negative_weight: 10
  val_every: 3
  seed: 1
  features_cache: {root}/features
  runs_dir: {root}/runs
postproc:
  threshold: 0.5
  patience_frames: 2
  refractory_seconds: 2.0
gating:
  preroll_frames: 50
  partial_chunk: drop
eval:
  fa_audio_dirs: []
  recall_floor_quiet: 0.95
  recall_floor_noisy: 0.80
  max_fa_per_hour: 0.5
  max_latency_ms: 500
"""


@pytest.fixture
def ww_cfg(tmp_path):
    cfg_path = tmp_path / "ww.yaml"
    cfg_path.write_text(WW_CONFIG_TEMPLATE.format(root=tmp_path))
    return load_ww_config(cfg_path)
