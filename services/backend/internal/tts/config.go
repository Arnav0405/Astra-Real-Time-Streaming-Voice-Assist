// Package tts streams synthesized speech from an OpenAI-compatible
// /audio/speech endpoint. Like llm, requests are context-cancellable so
// barge-in stops the provider mid-sentence instead of paying for audio nobody
// will hear.
package tts

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config is the runtime TTS policy, loaded from assets/configs/tts.json.
type Config struct {
	BaseURL string `json:"base_url"` // e.g. https://api.naga.ac/v1
	Model   string `json:"model"`
	Voice   string `json:"voice"`
	// Format is the provider's response_format. "pcm" means headerless mono
	// s16le, which is what the streaming path expects — anything else would
	// need decoding the server has no reason to do.
	Format string `json:"format"`
	// SampleRateHz is the rate Format implies for this provider. It is
	// declared rather than sniffed because headerless PCM carries no rate.
	// A wrong value is audible as wrong pitch on the first reply, so it fails
	// loudly rather than corrupting anything silently.
	//
	// ponytail: declared, not sniffed. If a provider ever streams WAV instead,
	// parse the fmt chunk here and drop this field.
	SampleRateHz int `json:"sample_rate_hz"`
}

// LoadConfig reads and validates the TTS config file.
func LoadConfig(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("tts config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Config{}, fmt.Errorf("tts config %s: %w", path, err)
	}
	if cfg.BaseURL == "" || cfg.Model == "" || cfg.Voice == "" {
		return Config{}, fmt.Errorf("tts config %s: base_url, model and voice are required", path)
	}
	if cfg.Format == "" {
		cfg.Format = "pcm"
	}
	if cfg.Format != "pcm" {
		return Config{}, fmt.Errorf("tts config %s: format %q unsupported, want \"pcm\"", path, cfg.Format)
	}
	if cfg.SampleRateHz <= 0 {
		return Config{}, fmt.Errorf("tts config %s: sample_rate_hz must be > 0", path)
	}
	return cfg, nil
}
