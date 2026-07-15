package server

import (
	"log"
	"time"
)

// Sink consumes the Frame stream of one session. Run is called in its own
// goroutine and must drain frames until the channel closes; Wait blocks until
// Run has finished. VAD (Phase 3) plugs in here.
type Sink interface {
	Run(frames <-chan Frame)
	Wait()
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
