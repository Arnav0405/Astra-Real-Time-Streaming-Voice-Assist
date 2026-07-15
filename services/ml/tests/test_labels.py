import csv
import json

import numpy as np
import soundfile as sf
import torch

from astra_ml.data import chime, libriparty
from astra_ml.data.libriparty import LibriPartyDataset, frame_labels, speech_intervals

TOY_SESSION = {
    "1272": [
        {"start": 0.5, "stop": 1.0, "file": "a.flac", "lvl": -15},
        {"start": 0.9, "stop": 1.5, "file": "b.flac", "lvl": -12},
    ],
    "2035": [{"start": 3.0, "stop": 3.5, "file": "c.flac", "lvl": -20}],
    "noises": [{"start": 2.0, "stop": 2.2, "file": "n.wav", "lvl": -30}],
    "background": {"start": 0, "stop": 4.0, "file": "bg.wav", "lvl": -50},
}


def test_speech_intervals_merges_and_ignores_noise():
    assert speech_intervals(TOY_SESSION) == [(0.5, 1.5), (3.0, 3.5)]


def test_frame_labels_center_rule():
    labels = frame_labels([(0.5, 1.5)], n_frames=100)  # 2 s @ 20 ms
    # frame 24 center = 0.49 s → 0; frame 25 center = 0.51 s → 1; frame 74 center = 1.49 → 1
    assert labels[24] == 0 and labels[25] == 1 and labels[74] == 1 and labels[75] == 0
    assert labels.sum() == 50


def test_frame_labels_offset():
    labels = frame_labels([(1.0, 2.0)], n_frames=50, offset_s=1.0)
    assert labels.all()  # all frame centers (1.01–1.99 s) inside the interval
    assert frame_labels([(1.0, 2.0)], n_frames=50, offset_s=2.0).sum() == 0


def test_dataset_windows_and_audio(tmp_path):
    sr = 16_000
    (tmp_path / "session_0").mkdir()
    sf.write(
        tmp_path / "session_0" / "session_0_mixture.wav",
        np.full(9 * sr, 0.5, np.float32),
        sr,
    )
    meta = {"session_0": {"9": [{"start": 0.0, "stop": 4.0}]}}
    (tmp_path / "train.json").write_text(json.dumps(meta))

    ds = LibriPartyDataset(tmp_path / "train.json", tmp_path, crop_frames=200)
    assert len(ds) == 2  # 9 s → two full 4 s windows
    pcm, labels = ds[0]
    assert pcm.shape == (200 * 320,) and labels.shape == (200,)
    assert labels.all()  # first window fully speech
    _, labels1 = ds[1]
    assert labels1.sum() == 0  # 4–8 s window is silence
    assert torch.allclose(pcm, torch.full((64000,), 0.5), atol=1e-4)  # 16-bit wav roundtrip


def _write_chunk(root, name, vote):
    (root / "chunks").mkdir(exist_ok=True)
    with open(root / "chunks" / f"{name}.csv", "w", newline="") as f:
        csv.writer(f).writerows([["segmentname", "x"], ["majorityvote", vote]])
    sf.write(root / "chunks" / f"{name}.16kHz.wav", np.zeros(64000, np.float32), 16_000)


def test_chime_classify_and_backgrounds(tmp_path):
    votes = {"c1": "cv", "c2": "b", "c3": "", "c4": "p", "c5": "m"}
    for name, vote in votes.items():
        _write_chunk(tmp_path, name, vote)
    speech, nonspeech = chime.classify(tmp_path, list(votes))
    assert speech == ["c1", "c5"]
    assert nonspeech == ["c2", "c4"]  # empty vote dropped

    n = chime.build_backgrounds(tmp_path, tmp_path / "bg", ["c2", "c4"] * 8)
    assert n == 1  # 16 chunks → one 60 s file (15 used)
    info = sf.info(tmp_path / "bg" / "chime_bg_000.wav")
    assert info.frames == 15 * 64000 and info.samplerate == libriparty.SAMPLE_RATE
