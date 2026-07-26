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
	BaseURL  string `json:"base_url"` // e.g. https://api.naga.ac/v1
	Model    string `json:"model"`    // e.g. whisper-large-v3:free
	Language string `json:"language"` // ISO 639-1 hint; empty = provider auto-detect
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
	return cfg, nil
}
