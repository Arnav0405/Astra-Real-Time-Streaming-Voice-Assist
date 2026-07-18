package wakeword

// Exact port of WwPostprocessor in services/ml/src/astra_ml/postproc_ww.py.
// Kept in lockstep with the Python reference; parity is enforced by
// testdata/ww_postproc_golden.json.
//
// A wake fires when the score stays at or above the threshold for
// patience_frames consecutive 80 ms score steps; a refractory window measured
// in absolute 20 ms frame indices then suppresses re-triggers, surviving VAD
// gate close/open cycles. gateReset (speech_end) clears only the run counter.

const never = -1 << 30

type postprocessor struct {
	cfg         *Config
	run         int
	lastTrigger int
}

func newPostprocessor(cfg *Config) *postprocessor {
	return &postprocessor{cfg: cfg, lastTrigger: never}
}

// push consumes one 80 ms score step; frame is the absolute index of the
// chunk's last 20 ms frame. Returns the trigger frame when a wake fires.
func (p *postprocessor) push(score float64, frame int) (int, bool) {
	if score >= p.cfg.Threshold {
		p.run++
	} else {
		p.run = 0
		return 0, false
	}
	if p.run < p.cfg.Postproc.PatienceFrames {
		return 0, false
	}
	if frame-p.lastTrigger < p.cfg.Postproc.RefractoryFrames {
		return 0, false
	}
	p.run = 0
	p.lastTrigger = frame
	return frame, true
}

// gateReset marks the score stream as no longer consecutive (VAD gate closed).
func (p *postprocessor) gateReset() {
	p.run = 0
}
