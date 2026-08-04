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

// modelVariants are graded independently: v1 (frozen OWW frontend + MLP head) and v2
// (log-mel + BC-ResNet) ship side by side until v2 clears its eval gates. Each variant
// skips until its artifacts and fixture exist, so an untrained v2 is not a failure.
var modelVariants = []struct {
	name    string
	model   string
	sidecar string
	fixture string
}{
	{"ww_v1", modelPath, sidecarPath, "testdata/ww_inference_golden.json"},
	{
		"ww_v2",
		"../../../../assets/models/wakeword/ww_v2.onnx",
		"../../../../assets/models/wakeword/ww_v2.json",
		"testdata/ww_inference_golden_v2.json",
	},
}

// TestInferenceGolden replays deterministic PCM through each committed merged model
// and requires score parity with the Python-generated fixture (1e-4).
func TestInferenceGolden(t *testing.T) {
	for _, v := range modelVariants {
		t.Run(v.name, func(t *testing.T) {
			data, err := os.ReadFile(v.fixture)
			if os.IsNotExist(err) {
				t.Skipf("%s not generated yet (needs committed %s artifacts)", v.fixture, v.name)
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(v.model); os.IsNotExist(err) {
				t.Skipf("%s.onnx not committed yet", v.name)
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

			cfg, err := LoadConfig(v.sidecar)
			if err != nil {
				t.Fatal(err)
			}
			if cfg.WindowSamples != g.WindowSamples {
				t.Fatalf("sidecar window %d != fixture %d", cfg.WindowSamples, g.WindowSamples)
			}
			engine, err := NewEngine(v.model, cfg)
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
		})
	}
}
