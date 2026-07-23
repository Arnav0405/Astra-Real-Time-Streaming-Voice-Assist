package endpoint

import (
	"testing"

	"github.com/arnav/astra/services/backend/internal/vad"
	"github.com/arnav/astra/services/backend/internal/wakeword"
)

const frameBytes = 640

// harness drives a Machine and records emitted utterances. seq is auto-advanced
// so frame indices stay ordered without threading a counter through each test.
type harness struct {
	m    *Machine
	seq  uint64
	utts []Utterance
}

func newHarness(mode Mode, cfg *Config) *harness {
	h := &harness{}
	h.m = NewMachine("test", cfg, mode, func(u Utterance) { h.utts = append(h.utts, u) })
	return h
}

// frame feeds one PCM frame (the real sink order: event first, then OnFrame).
func (h *harness) frame() {
	h.m.OnFrame(h.seq, make([]byte, frameBytes))
	h.seq++
}

func (h *harness) frames(n int) {
	for i := 0; i < n; i++ {
		h.frame()
	}
}

func (h *harness) vadStart() { h.m.OnVad(vad.Event{Type: vad.EventStart}) }
func (h *harness) vadEnd()   { h.m.OnVad(vad.Event{Type: vad.EventEnd}) }

func testCfg() *Config {
	return &Config{GraceFrames: 15, MinUtteranceFrames: 15, MaxUtteranceFrames: 1500}
}

// (a) normal close: speech then a full grace window of silence.
func TestNormalClose(t *testing.T) {
	h := newHarness(ArmOnVad, testCfg())
	h.vadStart()
	h.frames(20)
	h.vadEnd()
	h.frames(15) // grace expires on the 15th

	if len(h.utts) != 1 {
		t.Fatalf("want 1 utterance, got %d", len(h.utts))
	}
	if got := h.utts[0].FrameCount; got != 20 {
		t.Errorf("FrameCount = %d, want 20 (speech span, excl. grace tail)", got)
	}
	// buffer includes the grace tail: 20 + 15 frames.
	if got, want := len(h.utts[0].PCM), 35*frameBytes; got != want {
		t.Errorf("PCM len = %d, want %d", got, want)
	}
}

// (b) a pause shorter than grace reopens the same utterance — not two.
func TestGraceReopen(t *testing.T) {
	h := newHarness(ArmOnVad, testCfg())
	h.vadStart()
	h.frames(20)
	h.vadEnd()
	h.frames(5)  // pause < grace(15)
	h.vadStart() // speech resumes → reopen
	h.frames(20)
	h.vadEnd()
	h.frames(15) // now grace expires

	if len(h.utts) != 1 {
		t.Fatalf("want 1 utterance (reopened), got %d", len(h.utts))
	}
	if got := h.utts[0].FrameCount; got != 45 { // 20 + 5 + 20 at 2nd EventEnd
		t.Errorf("FrameCount = %d, want 45", got)
	}
}

// (c) a sub-floor blip is dropped: no callback.
func TestMinDrop(t *testing.T) {
	h := newHarness(ArmOnVad, testCfg())
	h.vadStart()
	h.frames(5) // < MinUtteranceFrames(15)
	h.vadEnd()
	h.frames(15)

	if len(h.utts) != 0 {
		t.Fatalf("want 0 utterances (dropped), got %d", len(h.utts))
	}
}

// (d) an utterance with no EventEnd force-closes at the max-timeout.
func TestMaxTimeout(t *testing.T) {
	cfg := &Config{GraceFrames: 15, MinUtteranceFrames: 15, MaxUtteranceFrames: 30}
	h := newHarness(ArmOnVad, cfg)
	h.vadStart()
	h.frames(30) // hits MaxUtteranceFrames with no EventEnd

	if len(h.utts) != 1 {
		t.Fatalf("want 1 utterance (timeout), got %d", len(h.utts))
	}
	if got := h.utts[0].FrameCount; got != 30 {
		t.Errorf("FrameCount = %d, want 30", got)
	}
}

// (e) mode gating: in wake mode a bare VAD EventStart must not arm; only the
// wake Event does.
func TestWakeModeArming(t *testing.T) {
	h := newHarness(ArmOnWake, testCfg())

	// VAD speech alone: nothing captured.
	h.vadStart()
	h.frames(20)
	h.vadEnd()
	h.frames(15)
	if len(h.utts) != 0 {
		t.Fatalf("wake mode: VAD start armed without a wake event, got %d utterances", len(h.utts))
	}

	// Now a wake event arms; speech is captured to the next EventEnd + grace.
	h.m.OnWake(wakeword.Event{})
	h.frames(20)
	h.vadEnd()
	h.frames(15)
	if len(h.utts) != 1 {
		t.Fatalf("wake mode: want 1 utterance after wake, got %d", len(h.utts))
	}
}
