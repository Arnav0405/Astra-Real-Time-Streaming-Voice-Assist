"""Fetch the Piper TTS voice into assets/models/tts/ (gitignored).

Idempotent: files already present are skipped. Run via `make tts-voice`.
"""

import urllib.request
from pathlib import Path

VOICE = "en_US-lessac-medium"
BASE = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/{VOICE}"
OUT = Path(__file__).resolve().parents[1] / "assets" / "models" / "tts"

OUT.mkdir(parents=True, exist_ok=True)
for suffix in (".onnx", ".onnx.json"):
    dest = OUT / (VOICE + suffix)
    if dest.exists():
        print(f"exists, skipping: {dest}")
        continue
    print(f"downloading {BASE}{suffix}")
    urllib.request.urlretrieve(f"{BASE}{suffix}", dest)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
