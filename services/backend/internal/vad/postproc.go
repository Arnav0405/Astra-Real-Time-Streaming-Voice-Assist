package vad

// Exact port of VadPostprocessor (services/ml/src/astra_ml/postproc.py).
// Keep both in lockstep; parity is enforced by postproc_golden.json.
//
// States: IDLE (onset run counting) and SPEECH (silence run counting).
// Entering speech requires MinSpeechFrames consecutive frames >= onset;
// leaving requires MinSilenceFrames consecutive frames < offset. Emitted
// frame indices are retroactive: "start" points at the first frame of the
// onset run, "end" at the first frame of the silence run (exclusive end).

type EventType string

const (
	EventStart EventType = "start"
	EventEnd   EventType = "end"
)

// Event marks a speech boundary at a Frame index (frame-counted, no wall-clock).
type Event struct {
	Type  EventType
	Frame int
}

const (
	idle = iota
	speech
)

type postprocessor struct {
	cfg      *Config
	state    int
	frame    int
	run      int // consecutive qualifying frames in current state
	runStart int // frame index where the current run began
}

func newPostprocessor(cfg *Config) *postprocessor {
	return &postprocessor{cfg: cfg}
}

func (p *postprocessor) push(prob float64) (Event, bool) {
	var event Event
	var ok bool
	if p.state == idle {
		if prob >= p.cfg.Onset {
			if p.run == 0 {
				p.runStart = p.frame
			}
			p.run++
			if p.run >= p.cfg.Postproc.MinSpeechFrames {
				p.state, p.run = speech, 0
				event, ok = Event{EventStart, p.runStart}, true
			}
		} else {
			p.run = 0
		}
	} else {
		if prob < p.cfg.Postproc.Offset {
			if p.run == 0 {
				p.runStart = p.frame
			}
			p.run++
			if p.run >= p.cfg.Postproc.MinSilenceFrames {
				p.state, p.run = idle, 0
				event, ok = Event{EventEnd, p.runStart}, true
			}
		} else {
			p.run = 0
		}
	}
	p.frame++
	return event, ok
}

// finish closes an open segment at end of stream.
func (p *postprocessor) finish() []Event {
	if p.state != speech {
		return nil
	}
	end := p.frame
	if p.run > 0 {
		end = p.runStart
	}
	p.state, p.run = idle, 0
	return []Event{{EventEnd, end}}
}
