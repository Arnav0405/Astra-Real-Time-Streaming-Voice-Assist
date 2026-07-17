package server

import (
	"log"
	"time"
)

// Sink consumes the Frame stream of one session. Run is called in its own
// goroutine and must drain frames until the channel closes (even after a
// fatal error, so the session never blocks); Wait blocks until Run has
// finished. Fatal is closed if the sink hits an unrecoverable error, after
// which Err reports it; the session then aborts the stream. The VAD sink
// (Phase 3) plugs in here.
type Sink interface {
	Run(frames <-chan Frame)
	Wait()
	Fatal() <-chan struct{}
	Err() error
}

// statsSink is the Phase 1 consumer: it counts what arrived and logs a
// summary when the stream ends.
type statsSink struct {
	streamID string
	done     chan struct{}

	frames   uint64
	bytes    uint64
	duration time.Duration
}

func newStatsSink(streamID string) *statsSink {
	return &statsSink{streamID: streamID, done: make(chan struct{})}
}

func (s *statsSink) Run(frames <-chan Frame) {
	defer close(s.done)
	for f := range frames {
		s.frames++
		s.bytes += uint64(len(f.PCM))
		s.duration += frameDurationMs * time.Millisecond
	}
	log.Printf("stream %s ended: %d frames, %d bytes, %s of audio",
		s.streamID, s.frames, s.bytes, s.duration)
}

func (s *statsSink) Wait() { <-s.done }

// statsSink never fails: a nil channel never fires.
func (s *statsSink) Fatal() <-chan struct{} { return nil }
func (s *statsSink) Err() error             { return nil }
