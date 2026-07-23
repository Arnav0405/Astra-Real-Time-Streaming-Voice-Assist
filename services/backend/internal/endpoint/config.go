// Package endpoint is the Phase 5 utterance layer. It ties a start trigger
// (wake Event, or VAD EventStart in VAD-only mode) to a subsequent VAD
// EventEnd, buffers the PCM of that span, and hands a closed Utterance to a
// server-side consumer (Phase 6 ASR; a WAV dumper for verification today).
//
// Endpointing reuses the VAD EventEnd (which already carries the ~680 ms
// trailing-silence hangover) and adds a short grace timer on top: after
// EventEnd, GraceFrames of continued silence closes the utterance, but a new
// EventStart inside the window reopens it — so a natural mid-turn pause does
// not cut the speaker off. All timing is frame-counted (20 ms/frame), no
// wall-clock, matching the frame-driven VAD/wake-word path.
package endpoint

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config holds the turn-taking policy (assets/configs/endpoint.json). These
// are runtime knobs, not model-coupled — they live in assets/configs, not the
// VAD/wake-word model sidecars.
type Config struct {
	// GraceFrames is the trailing-silence window after VAD EventEnd before an
	// utterance closes; a new EventStart within it reopens the utterance.
	GraceFrames int `json:"grace_frames"`
	// MinUtteranceFrames is the speech-span floor (arm..EventEnd). Shorter
	// utterances (false wake fire, cough) are dropped silently.
	MinUtteranceFrames int `json:"min_utterance_frames"`
	// MaxUtteranceFrames force-closes a stuck-open utterance, bounding the
	// buffer.
	MaxUtteranceFrames int `json:"max_utterance_frames"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("endpoint config: %w", err)
	}
	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("endpoint config %s: %w", path, err)
	}
	if c.GraceFrames <= 0 || c.MinUtteranceFrames <= 0 || c.MaxUtteranceFrames <= 0 {
		return nil, fmt.Errorf("endpoint config %s: grace/min/max frames must be > 0", path)
	}
	return &c, nil
}
