package vad

import (
	"encoding/binary"
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// DefaultLibPaths are probed in order when no -ort-lib flag / ASTRA_ORT_LIB
// env is given. Homebrew on macOS today; Linux path for the Docker phase.
var DefaultLibPaths = []string{
	"/opt/homebrew/lib/libonnxruntime.dylib",
	"/usr/local/lib/libonnxruntime.so",
}

// Init loads the ONNX Runtime shared library and initializes the environment.
// Call once at startup. libPath "" probes DefaultLibPaths.
func Init(libPath string) error {
	if libPath == "" {
		libPath = os.Getenv("ASTRA_ORT_LIB")
	}
	if libPath == "" {
		for _, p := range DefaultLibPaths {
			if _, err := os.Stat(p); err == nil {
				libPath = p
				break
			}
		}
	}
	if libPath == "" {
		return fmt.Errorf("onnxruntime library not found (tried %v); install with `brew install onnxruntime` or set -ort-lib / ASTRA_ORT_LIB", DefaultLibPaths)
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("onnxruntime init (%s): %w", libPath, err)
	}
	return nil
}

// Engine holds the loaded model. One per process; Run is thread-safe, so all
// streams share the session while each keeps its own GRU state.
type Engine struct {
	session *ort.DynamicAdvancedSession
	cfg     *Config
}

func NewEngine(modelPath string, cfg *Config) (*Engine, error) {
	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"pcm", "state_in"}, []string{"prob", "state_out"}, nil)
	if err != nil {
		return nil, fmt.Errorf("vad model %s: %w", modelPath, err)
	}
	return &Engine{session: session, cfg: cfg}, nil
}

func (e *Engine) Close() {
	e.session.Destroy()
}

// Inferencer is the per-stream inference state: preallocated tensors plus the
// GRU state threaded between frames (zeros at stream start).
type Inferencer struct {
	engine   *Engine
	pcm      *ort.Tensor[float32]
	stateIn  *ort.Tensor[float32]
	prob     *ort.Tensor[float32]
	stateOut *ort.Tensor[float32]
}

func (e *Engine) NewInferencer() (*Inferencer, error) {
	c := e.cfg
	pcm, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(c.FrameSamples)))
	if err != nil {
		return nil, err
	}
	stateIn, err := ort.NewEmptyTensor[float32](ort.NewShape(c.StateShape...))
	if err != nil {
		pcm.Destroy()
		return nil, err
	}
	prob, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		pcm.Destroy()
		stateIn.Destroy()
		return nil, err
	}
	stateOut, err := ort.NewEmptyTensor[float32](ort.NewShape(c.StateShape...))
	if err != nil {
		pcm.Destroy()
		stateIn.Destroy()
		prob.Destroy()
		return nil, err
	}
	return &Inferencer{engine: e, pcm: pcm, stateIn: stateIn, prob: prob, stateOut: stateOut}, nil
}

// Step runs one 20 ms frame of s16le PCM through the model and returns the
// speech probability, carrying the GRU state forward.
func (inf *Inferencer) Step(pcm []byte) (float64, error) {
	in := inf.pcm.GetData()
	if len(pcm) != len(in)*2 {
		return 0, fmt.Errorf("pcm must be %d bytes, got %d", len(in)*2, len(pcm))
	}
	for i := range in {
		in[i] = float32(int16(binary.LittleEndian.Uint16(pcm[i*2:]))) / 32768.0
	}
	err := inf.engine.session.Run(
		[]ort.Value{inf.pcm, inf.stateIn},
		[]ort.Value{inf.prob, inf.stateOut},
	)
	if err != nil {
		return 0, fmt.Errorf("vad inference: %w", err)
	}
	copy(inf.stateIn.GetData(), inf.stateOut.GetData())
	return float64(inf.prob.GetData()[0]), nil
}

func (inf *Inferencer) Close() {
	inf.pcm.Destroy()
	inf.stateIn.Destroy()
	inf.prob.Destroy()
	inf.stateOut.Destroy()
}
