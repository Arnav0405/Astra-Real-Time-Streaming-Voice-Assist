package wakeword

// scorer abstracts Inferencer for tests.
type scorer interface {
	Score(window []float32) (float64, error)
	Close()
}

// detector turns fed 20 ms frames into 80 ms score steps: it accumulates four
// frames into a 1280-sample chunk, shifts the scoring window left by one
// chunk, appends, and scores. The window starts (and resets to) all zeros —
// the cold-start behavior the Python reference (ww_eval.stream_scores) and the
// eval numbers assume.
type detector struct {
	sc     scorer
	window []float32 // WindowSamples, int16-range values
	chunk  []float32 // accumulator, cap ChunkSamples
}

func newDetector(sc scorer, cfg *Config) *detector {
	return &detector{
		sc:     sc,
		window: make([]float32, cfg.WindowSamples),
		chunk:  make([]float32, 0, cfg.ChunkSamples),
	}
}

// pushFrame feeds one 640-byte frame. When it completes a chunk, the window
// is advanced and scored; scored reports whether a score was produced.
func (d *detector) pushFrame(pcm []byte) (score float64, scored bool, err error) {
	d.chunk = pcmToFloats(d.chunk, pcm)
	if len(d.chunk) < cap(d.chunk) {
		return 0, false, nil
	}
	n := len(d.chunk)
	copy(d.window, d.window[n:])
	copy(d.window[len(d.window)-n:], d.chunk)
	d.chunk = d.chunk[:0]
	score, err = d.sc.Score(d.window)
	if err != nil {
		return 0, false, err
	}
	return score, true, nil
}

// reset drops any partial chunk and zeroes the window (VAD gate closed).
func (d *detector) reset() {
	d.chunk = d.chunk[:0]
	for i := range d.window {
		d.window[i] = 0
	}
}

func (d *detector) Close() {
	d.sc.Close()
}
