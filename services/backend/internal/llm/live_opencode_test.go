package llm

import (
	"context"
	"os"
	"strings"
	"testing"
)

// Live check against OpenCode Go — run manually, skipped without the key:
//
//	OPENCODE_GO_KEY=... go test ./internal/llm -run TestLiveOpenCodeGo -v
func TestLiveOpenCodeGo(t *testing.T) {
	key := os.Getenv("OPENCODE_GO_KEY")
	if key == "" {
		t.Skip("OPENCODE_GO_KEY not set")
	}
	cfg, err := LoadConfig("../../../../assets/configs/llm.json")
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if os.Getenv("ASTRA_LIVE_MODEL") != "" {
		cfg.Model = os.Getenv("ASTRA_LIVE_MODEL")
	}
	var got strings.Builder
	c := NewClient(cfg, key)
	if err := c.StreamWithHistory(context.Background(),
		"What is two plus two? Say only the number.",
		[]ChatMessage{{Role: "user", Content: "My name is Arnav."}},
		func(s string) { got.WriteString(s) }); err != nil {
		t.Fatalf("StreamWithHistory: %v", err)
	}
	t.Logf("model=%s reply=%q", cfg.Model, got.String())
	if got.Len() == 0 {
		t.Fatal("empty reply")
	}
}
