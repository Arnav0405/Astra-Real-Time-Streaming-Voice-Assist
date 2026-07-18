package vad

// Detector is the synchronous per-frame VAD API: one 20 ms frame in, at most
// one Event out. vad.Sink wraps it behind a channel for standalone use; the
// wake-word sink (Phase 4) calls it inline so frame processing and event
// delivery stay deterministically ordered.
type Detector struct {
	step stepper
	pp   *postprocessor
}

// NewDetector allocates a per-stream detector. Callers own Close.
func NewDetector(engine *Engine, cfg *Config) (*Detector, error) {
	inf, err := engine.NewInferencer()
	if err != nil {
		return nil, err
	}
	return &Detector{step: inf, pp: newPostprocessor(cfg)}, nil
}

// Push processes one 640-byte frame. On inference error the frame still
// advances the postproc clock — event indices stay aligned with audio time —
// and the error is returned for the caller's failure policy.
func (d *Detector) Push(pcm []byte) (Event, bool, error) {
	prob, err := d.step.Step(pcm)
	if err != nil {
		d.pp.frame++
		return Event{}, false, err
	}
	e, ok := d.pp.push(prob)
	return e, ok, nil
}

// Finish flushes the open segment at end of stream.
func (d *Detector) Finish() []Event { return d.pp.finish() }

func (d *Detector) Close() { d.step.Close() }
