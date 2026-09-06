// Package llm streams an assistant reply from an OpenAI-compatible chat
// completions API. Unlike asr, every request is context-cancellable end to
// end: barge-in must abort the in-flight HTTP body read, not merely stop
// consuming it, or the provider keeps generating tokens nobody will hear.
package llm

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config is the runtime LLM policy, loaded from assets/configs/llm.json.
type Config struct {
	BaseURL string `json:"base_url"` // e.g. https://api.naga.ac/v1
	Model   string `json:"model"`
	// SystemPrompt shapes the reply for speech: the text is spoken aloud, so
	// it must stay short and free of markdown the TTS would read out.
	SystemPrompt string `json:"system_prompt"`
	// MaxTokens caps a reply. A runaway answer is a barge-in the user has to
	// perform manually; bound it server-side instead.
	MaxTokens int `json:"max_tokens"`
	// HistoryMaxTokens caps the total tokens for conversation history sent
	// with each request. History is truncated from the oldest to fit.
	HistoryMaxTokens int `json:"history_max_tokens"`
	// HistoryMaxTurns limits the number of user/assistant turns to include
	// in history. Older turns are dropped first.
	HistoryMaxTurns int `json:"history_max_turns"`
}

// LoadConfig reads and validates the LLM config file.
func LoadConfig(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("llm config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Config{}, fmt.Errorf("llm config %s: %w", path, err)
	}
	if cfg.BaseURL == "" || cfg.Model == "" {
		return Config{}, fmt.Errorf("llm config %s: base_url and model are required", path)
	}
	if cfg.MaxTokens <= 0 {
		return Config{}, fmt.Errorf("llm config %s: max_tokens must be > 0", path)
	}
	if cfg.HistoryMaxTokens <= 0 {
		cfg.HistoryMaxTokens = 2000
	}
	if cfg.HistoryMaxTurns <= 0 {
		cfg.HistoryMaxTurns = 4
	}
	return cfg, nil
}
