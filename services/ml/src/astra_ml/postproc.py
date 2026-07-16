"""VAD post-processing: hysteresis + frame-count state machine over frame probabilities.

Reference implementation for the Go runtime (decision #10). The Go port must
match this behavior exactly; parameters ship in the model sidecar JSON
(assets/models/vad/vad_v1.json, "postproc" block). Keep both in lockstep.

States: IDLE (onset run counting) and SPEECH (silence run counting).
Entering speech requires min_speech_frames consecutive frames >= onset_threshold;
leaving requires min_silence_frames consecutive frames < offset_threshold.
Emitted frame indices are retroactive: "start" points at the first frame of the
onset run, "end" at the first frame of the silence run (exclusive end).
"""

from dataclasses import dataclass

import numpy as np

IDLE, SPEECH = 0, 1

Event = tuple[str, int]


@dataclass(frozen=True)
class PostprocConfig:
    onset_threshold: float
    offset_threshold: float
    min_speech_frames: int
    min_silence_frames: int

    @classmethod
    def from_sidecar(cls, sidecar: dict) -> "PostprocConfig":
        pp = sidecar["postproc"]
        return cls(
            onset_threshold=sidecar["recommended_threshold"],
            offset_threshold=pp["offset_threshold"],
            min_speech_frames=pp["min_speech_frames"],
            min_silence_frames=pp["min_silence_frames"],
        )


class VadPostprocessor:
    def __init__(self, cfg: PostprocConfig):
        self.cfg = cfg
        self.state = IDLE
        self.frame = 0
        self.run = 0  # consecutive qualifying frames in current state
        self.run_start = 0  # frame index where the current run began

    def push(self, prob: float) -> Event | None:
        event = None
        if self.state == IDLE:
            if prob >= self.cfg.onset_threshold:
                if self.run == 0:
                    self.run_start = self.frame
                self.run += 1
                if self.run >= self.cfg.min_speech_frames:
                    self.state, self.run = SPEECH, 0
                    event = ("start", self.run_start)
            else:
                self.run = 0
        else:
            if prob < self.cfg.offset_threshold:
                if self.run == 0:
                    self.run_start = self.frame
                self.run += 1
                if self.run >= self.cfg.min_silence_frames:
                    self.state, self.run = IDLE, 0
                    event = ("end", self.run_start)
            else:
                self.run = 0
        self.frame += 1
        return event

    def finish(self) -> list[Event]:
        """Close an open segment at end of stream."""
        if self.state != SPEECH:
            return []
        end = self.run_start if self.run > 0 else self.frame
        self.state, self.run = IDLE, 0
        return [("end", end)]


def segments(probs: np.ndarray, cfg: PostprocConfig) -> list[tuple[int, int]]:
    """Offline helper: run the streaming machine over probs, return (start, end) frame pairs."""
    pp = VadPostprocessor(cfg)
    events = [e for p in probs if (e := pp.push(float(p)))]
    events += pp.finish()
    starts = [i for kind, i in events if kind == "start"]
    ends = [i for kind, i in events if kind == "end"]
    return list(zip(starts, ends, strict=True))
