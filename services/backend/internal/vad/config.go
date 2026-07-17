// Package vad runs the Phase 2 ONNX VAD model over 20 ms Frames and turns
// per-frame speech probabilities into start/end events via a hysteresis state
// machine ported from services/ml/src/astra_ml/postproc.py (decision #10).
package vad

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config mirrors the model sidecar (assets/models/vad/vad_v1.json). The onset
// threshold is the sidecar's top-level recommended_threshold, matching
// PostprocConfig.from_sidecar in postproc.py.
type Config struct {
	SampleRate   int     `json:"sample_rate"`
	FrameSamples int     `json:"frame_samples"`
	StateShape   []int64 `json:"state_shape"`
	Onset        float64 `json:"recommended_threshold"`
	Postproc     struct {
		Offset           float64 `json:"offset_threshold"`
		MinSpeechFrames  int     `json:"min_speech_frames"`
		MinSilenceFrames int     `json:"min_silence_frames"`
	} `json:"postproc"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("vad sidecar: %w", err)
	}
	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("vad sidecar %s: %w", path, err)
	}
	if c.FrameSamples == 0 || c.Onset == 0 || len(c.StateShape) != 3 {
		return nil, fmt.Errorf("vad sidecar %s: missing required fields", path)
	}
	return &c, nil
}
