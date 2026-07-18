package wakeword

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/arnav/astra/services/backend/internal/vad"
)

type inferenceGolden struct {
	WindowSamples int       `json:"window_samples"`
	ChunkSamples  int       `json:"chunk_samples"`
	PCMBase64     string    `json:"pcm_s16le_base64"`
	Scores        []float64 `json:"scores"`
}

const (
	modelPath   = "../../../../assets/models/wakeword/ww_v1.onnx"
	sidecarPath = "../../../../assets/models/wakeword/ww_v1.json"
)

// TestInferenceGolden replays deterministic PCM through the committed merged
// model and requires score parity with the Python-generated fixture (1e-4).
func TestInferenceGolden(t *testing.T) {
	data, err := os.ReadFile("testdata/ww_inference_golden.json")
	if os.IsNotExist(err) {
		t.Skip("ww_inference_golden.json not generated yet (needs committed ww_v1 artifacts)")
	}
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("ww_v1.onnx not committed yet")
	}
	if err := vad.Init(""); err != nil {
		t.Skipf("onnxruntime unavailable: %v", err)
	}

	var g inferenceGolden
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}
	pcm, err := base64.StdEncoding.DecodeString(g.PCMBase64)
	if err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfig(sidecarPath)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.WindowSamples != g.WindowSamples {
		t.Fatalf("sidecar window %d != fixture %d", cfg.WindowSamples, g.WindowSamples)
	}
	engine, err := NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	inf, err := engine.NewInferencer()
	if err != nil {
		t.Fatal(err)
	}
	defer inf.Close()

	window := make([]float32, g.WindowSamples)
	for k, want := range g.Scores {
		copy(window, window[g.ChunkSamples:])
		tail := window[len(window)-g.ChunkSamples:]
		for i := range tail {
			off := (k*g.ChunkSamples + i) * 2
			tail[i] = float32(int16(binary.LittleEndian.Uint16(pcm[off:])))
		}
		got, err := inf.Score(window)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(got-want) > 1e-4 {
			t.Fatalf("chunk %d: score %g, want %g", k, got, want)
		}
	}
}
