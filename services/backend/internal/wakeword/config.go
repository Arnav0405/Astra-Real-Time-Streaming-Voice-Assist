// Package wakeword runs the Phase 4 merged OpenWakeWord model ("Astraa") over
// VAD-gated audio and turns per-chunk scores into wake events via a trigger
// machine ported from services/ml/src/astra_ml/postproc_ww.py (decision #10:
// the Python reference always leads, parity enforced by golden fixtures in
// testdata/).
package wakeword

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config mirrors the model sidecar (assets/models/wakeword/ww_v1.json).
type Config struct {
	SampleRate    int     `json:"sample_rate"`
	Graph         string  `json:"graph"`
	ChunkSamples  int     `json:"chunk_samples"`
	WindowSamples int     `json:"window_samples"`
	Threshold     float64 `json:"recommended_threshold"`
	Postproc      struct {
		PatienceFrames   int `json:"patience_frames"`
		RefractoryFrames int `json:"refractory_frames"`
	} `json:"postproc"`
	Gating struct {
		PrerollFrames int    `json:"preroll_frames"`
		PartialChunk  string `json:"partial_chunk"`
	} `json:"gating"`
	IO struct {
		Merged struct {
			Input  string `json:"input"`
			Output string `json:"output"`
		} `json:"merged"`
	} `json:"io"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("wakeword sidecar: %w", err)
	}
	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("wakeword sidecar %s: %w", path, err)
	}
	// ponytail: only the merged single-model layout is implemented; the chain
	// fallback never fired at export time. Implement chain mode here if it does.
	if c.Graph != "merged" {
		return nil, fmt.Errorf("wakeword sidecar %s: graph %q not supported (only \"merged\")", path, c.Graph)
	}
	if c.ChunkSamples == 0 || c.WindowSamples == 0 || c.Threshold == 0 {
		return nil, fmt.Errorf("wakeword sidecar %s: missing required fields", path)
	}
	if c.Gating.PartialChunk != "drop" {
		return nil, fmt.Errorf("wakeword sidecar %s: partial_chunk %q not supported", path, c.Gating.PartialChunk)
	}
	return &c, nil
}
