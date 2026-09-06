// Package asr sends closed utterances to a Whisper-compatible transcription
// API and hands the transcript to a downstream consumer (Phase 7's seam).
package asr

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config is the runtime ASR policy, loaded from assets/configs/asr.json.
type Config struct {
	GRPCAddress   string `json:"grpc_address"`
	Model         string `json:"model"`
	Language      string `json:"language"`
	ChunkFrames   int    `json:"chunk_frames"`
	OverlapFrames int    `json:"overlap_frames"`
	GraceFrames   int    `json:"grace_frames"`
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
	if cfg.Model == "" {
		return Config{}, fmt.Errorf("asr config %s: model is required", path)
	}
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
