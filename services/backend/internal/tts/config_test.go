package tts

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "tts.json")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadConfigParsesSharedSchema(t *testing.T) {
	cfg, err := LoadConfig(writeConfig(t, `{
		"grpc_address": "localhost:50052",
		"voice": "en_US-lessac-medium",
		"sample_rate_hz": 22050,
		"length_scale": 1.0,
		"noise_scale": 0.667,
		"noise_w": 0.333
	}`))
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	want := Config{
		GRPCAddress: "localhost:50052", Voice: "en_US-lessac-medium",
		SampleRateHz: 22050, LengthScale: 1.0, NoiseScale: 0.667, NoiseW: 0.333,
	}
	if cfg != want {
		t.Errorf("got %+v, want %+v", cfg, want)
	}
}

func TestLoadConfigRejectsBadFile(t *testing.T) {
	for name, body := range map[string]string{
		"missing address": `{"voice":"v","sample_rate_hz":22050,"length_scale":1,"noise_scale":0.667,"noise_w":0.333}`,
		"missing voice":   `{"grpc_address":"x:1","sample_rate_hz":22050,"length_scale":1,"noise_scale":0.667,"noise_w":0.333}`,
		"missing rate":    `{"grpc_address":"x:1","voice":"v","length_scale":1,"noise_scale":0.667,"noise_w":0.333}`,
		"bad prosody":     `{"grpc_address":"x:1","voice":"v","sample_rate_hz":22050,"length_scale":0,"noise_scale":0.667,"noise_w":0.333}`,
		"negative rate":   `{"grpc_address":"x:1","voice":"v","sample_rate_hz":-5,"length_scale":1,"noise_scale":0.667,"noise_w":0.333}`,
		"not json":        `{oops`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadConfig(writeConfig(t, body)); err == nil {
				t.Errorf("%s: want error, got nil", name)
			}
		})
	}
}

func TestLoadConfigMissingFile(t *testing.T) {
	_, err := LoadConfig(filepath.Join(t.TempDir(), "nope.json"))
	if err == nil {
		t.Fatal("want missing-file error, got nil")
	}
	// POSIX says "no such file or directory"; Windows says "The system cannot
	// find the file specified." Match on the stdlib sentinel instead.
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("want not-exist error, got %v", err)
	}
}
