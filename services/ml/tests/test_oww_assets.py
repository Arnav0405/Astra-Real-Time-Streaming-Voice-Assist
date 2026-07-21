"""load_acav guards every consumer against a partial/corrupt 16 GB download."""

import numpy as np
import pytest

from astra_ml.data.oww_assets import load_acav
from astra_ml.models.ww import EMB_DIM, HEAD_FRAMES


def test_load_acav_accepts_valid_file(tmp_path):
    path = tmp_path / "acav.npy"
    np.save(path, np.zeros((5, HEAD_FRAMES, EMB_DIM), dtype=np.float16))
    arr = load_acav(path)
    assert arr.shape == (5, HEAD_FRAMES, EMB_DIM)
    assert arr.dtype == np.float16


def test_load_acav_rejects_wrong_shape_and_dtype(tmp_path):
    path = tmp_path / "acav.npy"
    np.save(path, np.zeros((5, 8, 8), dtype=np.float16))
    with pytest.raises(ValueError, match="unexpected ACAV file"):
        load_acav(path)
    np.save(path, np.zeros((5, HEAD_FRAMES, EMB_DIM), dtype=np.float32))
    with pytest.raises(ValueError, match="unexpected ACAV file"):
        load_acav(path)


def test_load_acav_rejects_truncated_download(tmp_path):
    # what a killed curl leaves behind: a file exists() passes but np.load cannot map
    path = tmp_path / "acav.npy"
    np.save(path, np.zeros((100, HEAD_FRAMES, EMB_DIM), dtype=np.float16))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match="ACAV file"):
        load_acav(path)
