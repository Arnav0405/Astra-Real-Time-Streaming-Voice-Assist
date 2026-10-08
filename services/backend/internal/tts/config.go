package tts

import (
	"encoding/json"
	"fmt"
	"os"
)

type Config struct {
	GRPCAddress  string  `json:"grpc_address"`
	Voice        string  `json:"voice"`
	SampleRateHz int     `json:"sample_rate_hz"`
	LengthScale  float64 `json:"length_scale"`
	NoiseScale   float64 `json:"noise_scale"`
	NoiseW       float64 `json:"noise_w"`
}

func LoadConfig(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("tts config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Config{}, fmt.Errorf("tts config %s: %w", path, err)
	}
	if cfg.GRPCAddress == "" || cfg.Voice == "" {
		return Config{}, fmt.Errorf("tts config %s: grpc_address and voice are required", path)
	}
	if cfg.SampleRateHz <= 0 {
		return Config{}, fmt.Errorf("tts config %s: sample_rate_hz must be > 0", path)
	}
	for name, v := range map[string]float64{
		"length_scale": cfg.LengthScale,
		"noise_scale":  cfg.NoiseScale,
		"noise_w":      cfg.NoiseW,
	} {
		if v <= 0 {
			return Config{}, fmt.Errorf("tts config %s: %s must be > 0", path, name)
		}
	}
	return cfg, nil
}
