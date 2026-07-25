// Inference micro-benchmarks: wall-clock per VAD frame (20 ms) and per
// wake-word window score, using the same in-process ONNX Runtime path the
// server runs. Run: go test ./tests -run x -bench Infer -benchmem
package tests

import (
	"sync"
	"testing"

	"github.com/arnav/astra/services/backend/internal/vad"
	"github.com/arnav/astra/services/backend/internal/wakeword"
)

var (
	initOnce sync.Once
	initErr  error
)

func ortInit() error {
	initOnce.Do(func() { initErr = vad.Init("") })
	return initErr
}

func BenchmarkInferVAD(b *testing.B) {
	if err := ortInit(); err != nil {
		b.Skipf("onnxruntime unavailable: %v", err)
	}
	cfg, err := vad.LoadConfig("../../../assets/models/vad/vad_v1.json")
	if err != nil {
		b.Fatal(err)
	}
	eng, err := vad.NewEngine("../../../assets/models/vad/vad_v1.onnx", cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer eng.Close()
	inf, err := eng.NewInferencer()
	if err != nil {
		b.Fatal(err)
	}
	defer inf.Close()

	frame := make([]byte, cfg.FrameSamples*2) // 20 ms s16le
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := inf.Step(frame); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkInferWakeWord(b *testing.B) {
	if err := ortInit(); err != nil {
		b.Skipf("onnxruntime unavailable: %v", err)
	}
	cfg, err := wakeword.LoadConfig("../../../assets/models/wakeword/ww_v1.json")
	if err != nil {
		b.Fatal(err)
	}
	eng, err := wakeword.NewEngine("../../../assets/models/wakeword/ww_v1.onnx", cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer eng.Close()
	inf, err := eng.NewInferencer()
	if err != nil {
		b.Fatal(err)
	}
	defer inf.Close()

	window := make([]float32, cfg.WindowSamples)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := inf.Score(window); err != nil {
			b.Fatal(err)
		}
	}
}
