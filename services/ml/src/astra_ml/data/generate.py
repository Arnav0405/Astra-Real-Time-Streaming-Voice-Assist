"""Download sources and generate the LibriParty dataset (run on the GPU machine).

Steps (each skipped if its output already exists):
  1. Download + extract LibriSpeech train-clean-100/dev-clean/test-clean and OpenSLR-28.
  2. Sparse-clone the SpeechBrain LibriParty recipe.
  3. Run create_custom_dataset.py with our paths (CHiME backgrounds from data.chime).

Usage:
    uv run python -m astra_ml.data.chime --chime-root datasets/chime_home \
        --out datasets/chime_prepared
    uv run python -m astra_ml.data.generate --config configs/vad_v1.yaml
"""

import argparse
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from astra_ml.training.config import load_config

LIBRISPEECH_URLS = [
    f"https://www.openslr.org/resources/12/{name}.tar.gz"
    for name in ("train-clean-100", "dev-clean", "test-clean")
]
RIRS_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
RECIPE_REPO = "https://github.com/speechbrain/speechbrain"
RECIPE_PATH = "recipes/LibriParty/generate_dataset"


def _fetch(url: str, dest: Path) -> Path:
    archive = dest / url.rsplit("/", 1)[1]
    if not archive.exists():
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, archive)
    return archive


def download_sources(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for url in LIBRISPEECH_URLS:
        subset = url.rsplit("/", 1)[1].removesuffix(".tar.gz")
        if (dest / "LibriSpeech" / subset).exists():
            print(f"LibriSpeech/{subset} present, skipping")
            continue
        with tarfile.open(_fetch(url, dest)) as tar:
            tar.extractall(dest)
    if (dest / "RIRS_NOISES").exists():
        print("RIRS_NOISES present, skipping")
    else:
        with zipfile.ZipFile(_fetch(RIRS_URL, dest)) as zf:
            zf.extractall(dest)


def ensure_recipe(recipe_dir: Path) -> Path:
    if not (recipe_dir / RECIPE_PATH).exists():
        recipe_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                RECIPE_REPO,
                str(recipe_dir),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(recipe_dir), "sparse-checkout", "set", RECIPE_PATH],
            check=True,
        )
    return recipe_dir / RECIPE_PATH


def generate(config_path: Path) -> None:
    cfg = load_config(config_path).data
    ml_root = Path.cwd()
    download_sources(cfg.sources_root)
    recipe = ensure_recipe(cfg.recipe_dir)

    backgrounds = ml_root / cfg.chime_prepared / "backgrounds"
    if not backgrounds.exists():
        sys.exit(f"{backgrounds} missing — run astra_ml.data.chime first")

    out = ml_root / cfg.libriparty_out
    if (out / "metadata").exists():
        print(f"{out} already generated, skipping (delete to regenerate)")
        return
    subprocess.run(
        [
            sys.executable,
            "create_custom_dataset.py",
            "dataset.yaml",
            f"--librispeech_root={ml_root / cfg.sources_root / 'LibriSpeech'}",
            f"--rirs_noises_root={ml_root / cfg.sources_root / 'RIRS_NOISES'}",
            f"--backgrounds_root={backgrounds}",
            f"--out_folder={out}",
            f"--metadata_folder={out / 'metadata'}",
        ],
        cwd=recipe,
        check=True,
    )
    print(f"generated LibriParty dataset at {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vad_v1.yaml"))
    args = parser.parse_args()
    generate(args.config)


if __name__ == "__main__":
    main()
