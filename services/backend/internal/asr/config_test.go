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
		"grpc_address": "localhost:50051",
		"model": "small",
		"language": "en"
	}`))
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.GRPCAddress != "localhost:50051" || cfg.Model != "small" || cfg.Language != "en" {
		t.Errorf("cfg = %+v", cfg)
	}
}

func TestLoadConfigRejectsMissingFields(t *testing.T) {
	for name, content := range map[string]string{
		"no model": `{"grpc_address": "localhost:50051"}`,
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

func TestLoadConfigDefaults(t *testing.T) {
	cfg, err := LoadConfig(writeTemp(t, `{"model": "m"}`))
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.GRPCAddress != "localhost:50051" {
		t.Errorf("grpc_address default = %q, want localhost:50051", cfg.GRPCAddress)
	}
	if cfg.ChunkFrames != 150 {
		t.Errorf("chunk_frames default = %d, want 150", cfg.ChunkFrames)
	}
	if cfg.OverlapFrames != 50 {
		t.Errorf("overlap_frames default = %d, want 50", cfg.OverlapFrames)
	}
	if cfg.GraceFrames != 2 {
		t.Errorf("grace_frames default = %d, want 2", cfg.GraceFrames)
	}
}
