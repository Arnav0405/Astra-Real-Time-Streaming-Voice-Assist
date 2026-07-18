package vad

import (
	"fmt"
	"log"

	"github.com/arnav/astra/services/backend/internal/server"
)

// maxConsecutiveFailures: single inference errors are logged and skipped, but
// this many in a row (~1 s of audio) means systemic breakage — the stream is
// killed instead of silently serving without VAD.
const maxConsecutiveFailures = 50

// stepper abstracts Inferencer for tests.
type stepper interface {
	Step(pcm []byte) (float64, error)
	Close()
}

// Sink consumes one session's Frames, runs VAD inference per frame, and emits
// speech start/end Events through onEvent. It implements server.Sink.
type Sink struct {
	streamID string
	newStep  func() (stepper, error)
	pp       *postprocessor
	onEvent  func(Event)

	done  chan struct{}
	fatal chan struct{}
	err   error // set before fatal is closed; read after Fatal() or Wait()
}

// NewSink returns a VAD sink for one stream. onEvent nil means log events.
func NewSink(streamID string, engine *Engine, cfg *Config, onEvent func(Event)) *Sink {
	if onEvent == nil {
		onEvent = func(e Event) {
			log.Printf("stream %s: vad %s at frame %d", streamID, e.Type, e.Frame)
		}
	}
	return &Sink{
		streamID: streamID,
		newStep:  func() (stepper, error) { s, err := engine.NewInferencer(); return s, err },
		pp:       newPostprocessor(cfg),
		onEvent:  onEvent,
		done:     make(chan struct{}),
		fatal:    make(chan struct{}),
	}
}

func (s *Sink) Run(frames <-chan server.Frame) {
	defer close(s.done)

	inf, err := s.newStep()
	if err != nil {
		s.fail(fmt.Errorf("vad init: %w", err))
		for range frames { // keep draining so the session never blocks
		}
		return
	}
	det := &Detector{step: inf, pp: s.pp}
	defer det.Close()

	failures := 0
	for f := range frames {
		if s.err != nil {
			continue // fatal already signaled; just drain
		}
		e, ok, err := det.Push(f.PCM)
		if err != nil {
			log.Printf("stream %s: frame %d: %v (skipped)", s.streamID, f.Seq, err)
			failures++
			if failures >= maxConsecutiveFailures {
				s.fail(fmt.Errorf("vad inference failed %d frames in a row: %w", failures, err))
			}
			continue
		}
		failures = 0
		if ok {
			s.onEvent(e)
		}
	}
	if s.err == nil {
		for _, e := range det.Finish() {
			s.onEvent(e)
		}
	}
}

func (s *Sink) fail(err error) {
	s.err = err
	close(s.fatal)
}

func (s *Sink) Wait()                  { <-s.done }
func (s *Sink) Fatal() <-chan struct{} { return s.fatal }
func (s *Sink) Err() error             { return s.err }
