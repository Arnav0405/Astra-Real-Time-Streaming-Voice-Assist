"""Wake-word trigger post-processing: raw chunk scores -> wake events.

Reference implementation for the Go runtime port (decision #10 pattern: the
Python side always leads, parity enforced by golden fixtures). A wake fires
when the score stays at or above `threshold` for `patience_frames` consecutive
80 ms score steps, then a refractory window suppresses re-triggers.

Frame bookkeeping is in absolute 20 ms transport-frame indices, not score
steps: `push(score, frame)` takes the index of the last 20 ms frame of the
scored chunk, and refractory compares absolute frame distance so it survives
VAD gate close/open cycles. `gate_reset()` (called on VAD speech_end) clears
only the consecutive-run counter — never the refractory clock.
"""

from dataclasses import dataclass

_NEVER = -(10**9)


@dataclass(frozen=True)
class WwPostprocConfig:
    threshold: float
    patience_frames: int  # consecutive 80 ms score steps at/above threshold
    refractory_frames: int  # absolute 20 ms frames between triggers

    @classmethod
    def from_sidecar(cls, sidecar: dict) -> "WwPostprocConfig":
        pp = sidecar["postproc"]
        return cls(
            threshold=sidecar["recommended_threshold"],
            patience_frames=pp["patience_frames"],
            refractory_frames=pp["refractory_frames"],
        )


class WwPostprocessor:
    def __init__(self, cfg: WwPostprocConfig):
        self.cfg = cfg
        self.run = 0
        self.last_trigger = _NEVER

    def push(self, score: float, frame: int) -> int | None:
        """One 80 ms score step; returns the trigger frame index or None."""
        if score >= self.cfg.threshold:
            self.run += 1
        else:
            self.run = 0
            return None
        if self.run < self.cfg.patience_frames:
            return None
        if frame - self.last_trigger < self.cfg.refractory_frames:
            return None
        self.run = 0
        self.last_trigger = frame
        return frame

    def gate_reset(self) -> None:
        """VAD gate closed: scores are no longer consecutive."""
        self.run = 0
