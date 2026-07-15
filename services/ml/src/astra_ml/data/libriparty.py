"""LibriParty metadata → per-20 ms frame labels, and a Dataset over generated mixtures.

Metadata JSON (written by SpeechBrain's create_custom_dataset.py):
{"session_0": {"<speaker_id>": [{"start": s, "stop": e, ...}, ...],
               "noises": [...], "background": {...}}, ...}
start/stop are seconds. Speech = union of all speaker intervals.
"""

import json
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from astra_ml.audio.dft_mel import FRAME_SAMPLES, SAMPLE_RATE

FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
NON_SPEAKER_KEYS = {"noises", "background"}


def speech_intervals(session: dict) -> list[tuple[float, float]]:
    """Merged (start, stop) speech intervals in seconds for one session."""
    raw = sorted(
        (float(utt["start"]), float(utt["stop"]))
        for key, utts in session.items()
        if key not in NON_SPEAKER_KEYS
        for utt in utts
    )
    merged: list[tuple[float, float]] = []
    for start, stop in raw:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def parse_metadata(path: Path) -> dict[str, list[tuple[float, float]]]:
    data = json.loads(Path(path).read_text())
    return {name: speech_intervals(session) for name, session in data.items()}


def frame_labels(
    intervals: list[tuple[float, float]], n_frames: int, offset_s: float = 0.0
) -> torch.Tensor:
    """Binary [n_frames] labels: 1 if the frame center falls inside a speech interval."""
    labels = torch.zeros(n_frames)
    centers = offset_s + (torch.arange(n_frames) + 0.5) * FRAME_SECONDS
    for start, stop in intervals:
        labels[(centers >= start) & (centers < stop)] = 1.0
    return labels


def find_mixtures(root: Path) -> dict[str, Path]:
    """{"session_0": .../session_0_mixture.wav} for all mixtures under root."""
    return {p.name.removesuffix("_mixture.wav"): p for p in Path(root).rglob("*_mixture.wav")}


class LibriPartyDataset(Dataset):
    """Non-overlapping crops of `crop_frames` frames → (pcm [crop*320], labels [crop])."""

    def __init__(self, metadata_json: Path, mixtures_root: Path, crop_frames: int = 200) -> None:
        self.crop_frames = crop_frames
        sessions = parse_metadata(metadata_json)
        mixtures = find_mixtures(mixtures_root)
        missing = sessions.keys() - mixtures.keys()
        if missing:
            raise FileNotFoundError(f"no mixture wav for sessions: {sorted(missing)[:5]} ...")
        self.windows: list[tuple[Path, int, torch.Tensor]] = []
        for name, intervals in sessions.items():
            wav = mixtures[name]
            total_frames = sf.info(wav).frames // FRAME_SAMPLES
            for w in range(total_frames // crop_frames):
                start_frame = w * crop_frames
                labels = frame_labels(intervals, crop_frames, start_frame * FRAME_SECONDS)
                self.windows.append((wav, start_frame, labels))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav, start_frame, labels = self.windows[idx]
        pcm, _ = sf.read(
            wav,
            start=start_frame * FRAME_SAMPLES,
            frames=self.crop_frames * FRAME_SAMPLES,
            dtype="float32",
            always_2d=True,
        )
        return torch.from_numpy(pcm[:, 0]), labels
