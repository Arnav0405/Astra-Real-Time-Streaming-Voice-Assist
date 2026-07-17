package vad

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestInferenceGolden runs the committed model over the Python-generated PCM
// and requires per-frame probability parity with onnxruntime-python within
// 1e-4. Skips when no ONNX Runtime library is available.
func TestInferenceGolden(t *testing.T) {
	if err := Init(""); err != nil {
		t.Skipf("onnxruntime unavailable: %v", err)
	}

	data, err := os.ReadFile("testdata/inference_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		PCM          string    `json:"pcm_s16le_base64"`
		FrameSamples int       `json:"frame_samples"`
		Probs        []float64 `json:"probs"`
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	pcm, err := base64.StdEncoding.DecodeString(golden.PCM)
	if err != nil {
		t.Fatal(err)
	}

	cfg := loadConfigT(t)
	engine, err := NewEngine("../../../../assets/models/vad/vad_v1.onnx", cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	inf, err := engine.NewInferencer()
	if err != nil {
		t.Fatal(err)
	}
	defer inf.Close()

	frameBytes := golden.FrameSamples * 2
	for i, want := range golden.Probs {
		got, err := inf.Step(pcm[i*frameBytes : (i+1)*frameBytes])
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		if math.Abs(got-want) > 1e-4 {
			t.Fatalf("frame %d: prob %.6f, python %.6f (diff %.2g)", i, got, want, math.Abs(got-want))
		}
	}
}
