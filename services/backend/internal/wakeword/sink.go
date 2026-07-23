package wakeword

import (
	"fmt"
	"log"

	"github.com/arnav/astra/services/backend/internal/server"
	"github.com/arnav/astra/services/backend/internal/vad"
)

// maxConsecutiveFailures mirrors the vad sink policy: single inference errors
// are logged and skipped, this many in a row kills the stream.
const maxConsecutiveFailures = 50

// Event is a wake-word detection. Frame is the absolute 20 ms frame index of
// the last frame of the scored chunk that fired. Downstream (Phase 5) anchors
// utterance capture on the VAD speech_start, not on this index.
type Event struct {
	Frame int
}

// vadDetector abstracts vad.Detector for tests.
type vadDetector interface {
	Push(pcm []byte) (vad.Event, bool, error)
	Finish() []vad.Event
	Close()
}

// Sink runs VAD synchronously per frame and gates the wake-word detector on
// speech: frames are scored only between speech_start and speech_end, with a
// ring-buffer backfill covering the retroactive start plus preroll_frames of
// left context (the scoring window is zero-initialized at gate open, so
// preroll fills it with real audio). It implements server.Sink and is the
// Phase 4 replacement for vad.Sink at the Server.NewSink seam.
//
// The gating semantics are pinned by the Python reference (GatingSim in
// services/ml/src/astra_ml/export/golden_ww.py) via testdata/ww_gating_golden.json:
//   - speech_start delivered at frame i with retroactive event frame f:
//     backfill frames max(0, f-preroll) .. i inclusive, then keep feeding
//   - every 4th fed frame completes an 80 ms chunk -> score -> trigger machine
//     (frame index = the chunk's last frame)
//   - speech_end at frame i: frame i is NOT fed; the partial chunk is dropped,
//     the window resets, the trigger run resets (refractory clock survives)
type Sink struct {
	streamID string
	newParts func() (vadDetector, *detector, error)
	cfg      *Config
	ringSize int
	onVad    func(vad.Event)
	onWake   func(Event)
	onFrame  func(uint64, []byte)

	done  chan struct{}
	fatal chan struct{}
	err   error // set before fatal is closed; read after Fatal() or Wait()
}

// NewSink returns a wake-word sink for one stream. Nil callbacks log.
// onFrame, if non-nil, is called once per frame with its seq and PCM (after any
// event for that frame) — the seam the Phase 5 endpoint machine buffers from.
func NewSink(streamID string, vadEngine *vad.Engine, vadCfg *vad.Config,
	engine *Engine, cfg *Config, onVad func(vad.Event), onWake func(Event), onFrame func(uint64, []byte)) *Sink {
	if onVad == nil {
		onVad = func(e vad.Event) {
			log.Printf("stream %s: vad %s at frame %d", streamID, e.Type, e.Frame)
		}
	}
	if onWake == nil {
		onWake = func(e Event) {
			log.Printf("stream %s: wake at frame %d", streamID, e.Frame)
		}
	}
	if onFrame == nil {
		onFrame = func(uint64, []byte) {}
	}
	return &Sink{
		streamID: streamID,
		newParts: func() (vadDetector, *detector, error) {
			vd, err := vad.NewDetector(vadEngine, vadCfg)
			if err != nil {
				return nil, nil, err
			}
			inf, err := engine.NewInferencer()
			if err != nil {
				vd.Close()
				return nil, nil, err
			}
			return vd, newDetector(inf, cfg), nil
		},
		cfg: cfg,
		// Depth: retroactive speech_start reaches back min_speech_frames-1,
		// plus preroll_frames of context, plus slack.
		ringSize: cfg.Gating.PrerollFrames + vadCfg.Postproc.MinSpeechFrames + 8,
		onVad:    onVad,
		onWake:   onWake,
		onFrame:  onFrame,
		done:     make(chan struct{}),
		fatal:    make(chan struct{}),
	}
}

func (s *Sink) Run(frames <-chan server.Frame) {
	defer close(s.done)

	vd, det, err := s.newParts()
	if err != nil {
		s.fail(fmt.Errorf("wakeword init: %w", err))
		for range frames { // keep draining so the session never blocks
		}
		return
	}
	defer vd.Close()
	defer det.Close()

	ring := make([][]byte, s.ringSize)
	pp := newPostprocessor(s.cfg)
	gate := false
	frameIdx := -1
	vadFailures, wwFailures := 0, 0 // separate: a healthy VAD must not mask a broken scorer

	countFailure := func(counter *int, err error, what string, seq uint64) {
		log.Printf("stream %s: frame %d: %s: %v (skipped)", s.streamID, seq, what, err)
		*counter++
		if *counter >= maxConsecutiveFailures {
			s.fail(fmt.Errorf("%s failed %d times in a row: %w", what, *counter, err))
		}
	}

	feed := func(idx int, seq uint64) {
		score, scored, err := det.pushFrame(ring[idx%s.ringSize])
		if err != nil {
			countFailure(&wwFailures, err, "wakeword inference", seq)
			return
		}
		if !scored {
			return
		}
		wwFailures = 0
		if frame, ok := pp.push(score, idx); ok {
			s.onWake(Event{Frame: frame})
		}
	}

	for f := range frames {
		if s.err != nil {
			continue // fatal already signaled; just drain
		}
		frameIdx++
		slot := frameIdx % s.ringSize
		if ring[slot] == nil {
			ring[slot] = make([]byte, len(f.PCM))
		}
		copy(ring[slot], f.PCM)

		ev, ok, err := vd.Push(f.PCM)
		if err != nil {
			countFailure(&vadFailures, err, "vad inference", f.Seq)
			continue
		}
		vadFailures = 0

		switch {
		case ok && ev.Type == vad.EventStart:
			s.onVad(ev)
			gate = true
			start := ev.Frame - s.cfg.Gating.PrerollFrames
			if start < 0 {
				start = 0
			}
			for j := start; j <= frameIdx; j++ {
				feed(j, f.Seq)
			}
		case ok && ev.Type == vad.EventEnd:
			s.onVad(ev)
			gate = false
			det.reset()
			pp.gateReset()
		case gate:
			feed(frameIdx, f.Seq)
		}
		if s.onFrame != nil {
			s.onFrame(f.Seq, f.PCM)
		}
	}
	if s.err == nil {
		for _, e := range vd.Finish() {
			s.onVad(e)
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
