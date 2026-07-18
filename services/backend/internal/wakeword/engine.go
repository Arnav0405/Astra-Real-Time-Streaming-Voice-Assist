package wakeword

import (
	"encoding/binary"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// Engine holds the merged wake-word model (audio window in, probability out).
// One per process, shared across streams; vad.Init must have been called
// first — it owns the process-global ONNX Runtime environment.
type Engine struct {
	session *ort.DynamicAdvancedSession
	cfg     *Config
}

func NewEngine(modelPath string, cfg *Config) (*Engine, error) {
	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{cfg.IO.Merged.Input}, []string{cfg.IO.Merged.Output}, nil)
	if err != nil {
		return nil, fmt.Errorf("wakeword model %s: %w", modelPath, err)
	}
	return &Engine{session: session, cfg: cfg}, nil
}

func (e *Engine) Close() {
	e.session.Destroy()
}

// Inferencer is per-stream inference state: preallocated audio/prob tensors.
type Inferencer struct {
	engine *Engine
	audio  *ort.Tensor[float32]
	prob   *ort.Tensor[float32]
}

func (e *Engine) NewInferencer() (*Inferencer, error) {
	audio, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(e.cfg.WindowSamples)))
	if err != nil {
		return nil, err
	}
	prob, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		audio.Destroy()
		return nil, err
	}
	return &Inferencer{engine: e, audio: audio, prob: prob}, nil
}

// Score runs one full audio window (raw int16 sample values as float32 — the
// sidecar's input_scale contract, NOT [-1,1] like the VAD model) and returns
// the wake probability.
func (inf *Inferencer) Score(window []float32) (float64, error) {
	in := inf.audio.GetData()
	if len(window) != len(in) {
		return 0, fmt.Errorf("window must be %d samples, got %d", len(in), len(window))
	}
	copy(in, window)
	err := inf.engine.session.Run(
		[]ort.Value{inf.audio},
		[]ort.Value{inf.prob},
	)
	if err != nil {
		return 0, fmt.Errorf("wakeword inference: %w", err)
	}
	return float64(inf.prob.GetData()[0]), nil
}

func (inf *Inferencer) Close() {
	inf.audio.Destroy()
	inf.prob.Destroy()
}

var _ scorer = (*Inferencer)(nil)

// pcmToFloats appends the 320 s16le samples of one frame as raw int16 values.
func pcmToFloats(dst []float32, pcm []byte) []float32 {
	for i := 0; i+1 < len(pcm); i += 2 {
		dst = append(dst, float32(int16(binary.LittleEndian.Uint16(pcm[i:]))))
	}
	return dst
}
