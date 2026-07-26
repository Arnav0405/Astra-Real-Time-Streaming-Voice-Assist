package asr

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "asr.json")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadConfig(t *testing.T) {
	cfg, err := LoadConfig(writeTemp(t, `{
		"base_url": "https://api.naga.ac/v1",
		"model": "whisper-large-v3:free",
		"language": "en"
	}`))
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.BaseURL != "https://api.naga.ac/v1" || cfg.Model != "whisper-large-v3:free" || cfg.Language != "en" {
		t.Errorf("cfg = %+v", cfg)
	}
}

func TestLoadConfigRejectsMissingFields(t *testing.T) {
	for name, content := range map[string]string{
		"no base_url": `{"model": "m"}`,
		"no model":    `{"base_url": "https://x"}`,
	} {
		if _, err := LoadConfig(writeTemp(t, content)); err == nil {
			t.Errorf("%s: want error, got nil", name)
		}
	}
}

func TestLoadConfigMissingFile(t *testing.T) {
	if _, err := LoadConfig("/nonexistent/asr.json"); err == nil {
		t.Error("want error for missing file")
	}
}
