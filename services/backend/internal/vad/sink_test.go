package vad

import (
	"errors"
	"testing"

	"github.com/arnav/astra/services/backend/internal/server"
)

// fakeStepper returns scripted probs, or an error when prob < 0.
type fakeStepper struct {
	probs []float64
	i     int
}

func (f *fakeStepper) Step(pcm []byte) (float64, error) {
	p := f.probs[f.i%len(f.probs)]
	f.i++
	if p < 0 {
		return 0, errors.New("boom")
	}
	return p, nil
}
func (f *fakeStepper) Close() {}

func testSink(cfg *Config, fake *fakeStepper) (*Sink, *[]Event) {
	events := &[]Event{}
	s := &Sink{
		streamID: "test",
		newStep:  func() (stepper, error) { return fake, nil },
		pp:       newPostprocessor(cfg),
		onEvent:  func(e Event) { *events = append(*events, e) },
		done:     make(chan struct{}),
		fatal:    make(chan struct{}),
	}
	return s, events
}

func feed(s *Sink, n int) {
	frames := make(chan server.Frame, n)
	for i := 0; i < n; i++ {
		frames <- server.Frame{Seq: uint64(i), PCM: make([]byte, 640)}
	}
	close(frames)
	go s.Run(frames)
	s.Wait()
}

func TestSinkEmitsEvents(t *testing.T) {
	cfg := loadConfigT(t)
	// min_speech frames above onset, then min_silence below offset.
	probs := make([]float64, 0)
	for i := 0; i < cfg.Postproc.MinSpeechFrames; i++ {
		probs = append(probs, cfg.Onset+0.1)
	}
	for i := 0; i < cfg.Postproc.MinSilenceFrames; i++ {
		probs = append(probs, 0.0)
	}
	s, events := testSink(cfg, &fakeStepper{probs: probs})
	feed(s, len(probs))

	want := []Event{
		{EventStart, 0},
		{EventEnd, cfg.Postproc.MinSpeechFrames},
	}
	if len(*events) != len(want) {
		t.Fatalf("got %v, want %v", *events, want)
	}
	for i := range want {
		if (*events)[i] != want[i] {
			t.Fatalf("got %v, want %v", *events, want)
		}
	}
	if s.Err() != nil {
		t.Fatalf("unexpected fatal: %v", s.Err())
	}
}

func TestSinkSkipsTransientErrors(t *testing.T) {
	cfg := loadConfigT(t)
	// A few scattered errors must not kill the stream or emit events.
	s, events := testSink(cfg, &fakeStepper{probs: []float64{0.0, -1, 0.0, -1}})
	feed(s, 40)
	if s.Err() != nil {
		t.Fatalf("transient errors should not be fatal: %v", s.Err())
	}
	if len(*events) != 0 {
		t.Fatalf("unexpected events: %v", *events)
	}
}

func TestSinkFatalAfterConsecutiveFailures(t *testing.T) {
	cfg := loadConfigT(t)
	s, _ := testSink(cfg, &fakeStepper{probs: []float64{-1}})
	feed(s, maxConsecutiveFailures+10) // must keep draining past the cap

	select {
	case <-s.Fatal():
	default:
		t.Fatal("Fatal channel not closed")
	}
	if s.Err() == nil {
		t.Fatal("Err() should be set after fatal")
	}
}
