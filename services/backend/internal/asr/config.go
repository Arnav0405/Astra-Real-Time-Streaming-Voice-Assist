// Package asr sends closed utterances to a Whisper-compatible transcription
// API and hands the transcript to a downstream consumer (Phase 7's seam).
package asr

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config is the runtime ASR policy, loaded from assets/configs/asr.json.
// Timeout/retry/queue-size are code constants until they need tuning.
type Config struct {
	BaseURL        string `json:"base_url"`         // e.g. https://api.naga.ac/v1
	GRPCAddress    string `json:"grpc_address"`     // e.g. localhost:50051
	Model          string `json:"model"`            // e.g. whisper-large-v3:free
	Language       string `json:"language"`         // ISO 639-1 hint; empty = provider auto-detect
	ChunkFrames    int    `json:"chunk_frames"`     // NEW: 150 frames (3s at 20ms/frame)
	OverlapFrames  int    `json:"overlap_frames"`   // NEW: 50 frames (1s overlap)
	GraceFrames    int    `json:"grace_frames"`     // NEW: grace frames for endpointing
}

// LoadConfig reads and validates the ASR config file.
func LoadConfig(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("asr config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Config{}, fmt.Errorf("asr config %s: %w", path, err)
	}
	if cfg.BaseURL == "" || cfg.Model == "" {
		return Config{}, fmt.Errorf("asr config %s: base_url and model are required", path)
	}
	// Set defaults for optional fields
	if cfg.GRPCAddress == "" {
		cfg.GRPCAddress = "localhost:50051"
	}
	if cfg.ChunkFrames <= 0 {
		cfg.ChunkFrames = 150
	}
	if cfg.OverlapFrames <= 0 {
		cfg.OverlapFrames = 50
	}
	if cfg.GraceFrames <= 0 {
		cfg.GraceFrames = 2
	}
	return cfg, nil
}
