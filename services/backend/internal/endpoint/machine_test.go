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
	m      *Machine
	seq    uint64
	utts   []Utterance
	barges int
	arms   []armEvent
}

type armEvent struct {
	startSeq uint64
	preroll  []byte
}

func newHarness(mode Mode, cfg *Config) *harness {
	h := &harness{}
	h.m = NewMachine("test", cfg, mode,
		func(u Utterance) { h.utts = append(h.utts, u) },
		func() { h.barges++ },
		func(startSeq uint64, preroll []byte) {
			h.arms = append(h.arms, armEvent{startSeq: startSeq, preroll: append([]byte(nil), preroll...)})
		},
		nil)
	return h
}

// frame feeds one PCM frame (the real sink order: event first, then OnFrame).
// The PCM is stamped with the frame's seq so preroll tests can tell which
// frames actually made it into an utterance.
func (h *harness) frame() {
	pcm := make([]byte, frameBytes)
	for i := range pcm {
		pcm[i] = byte(h.seq)
	}
	h.m.OnFrame(h.seq, pcm)
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
	return &Config{GraceFrames: 15, MinUtteranceFrames: 15, MaxUtteranceFrames: 1500, BargeInFrames: 6}
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

// --- Barge-in (Phase 7b) ---------------------------------------------------

// Sustained speech during playback interrupts the assistant and opens an
// utterance, even in ArmOnWake mode where a bare speech_start would normally
// be ignored.
func TestBargeInFiresAfterThreshold(t *testing.T) {
	h := newHarness(ArmOnWake, testCfg())
	h.m.SetSpeaking(true)

	h.vadStart()
	h.frames(5) // one short of the 6-frame threshold
	if h.barges != 0 {
		t.Fatalf("barged after %d frames, want no fire before the threshold", 5)
	}
	h.frame() // the 6th confirms it
	if h.barges != 1 {
		t.Fatalf("barges = %d after reaching the threshold, want 1", h.barges)
	}

	// The utterance is now open and closes normally.
	h.frames(20)
	h.vadEnd()
	h.frames(15)
	if len(h.utts) != 1 {
		t.Fatalf("want 1 utterance after barge-in, got %d", len(h.utts))
	}
}

// A cough or a burst of echo shorter than the threshold must not cut the
// assistant off.
func TestShortBlipDoesNotBargeIn(t *testing.T) {
	h := newHarness(ArmOnWake, testCfg())
	h.m.SetSpeaking(true)

	h.vadStart()
	h.frames(3)
	h.vadEnd()
	h.frames(30)

	if h.barges != 0 {
		t.Errorf("barges = %d, want 0 for a blip below the threshold", h.barges)
	}
	if len(h.utts) != 0 {
		t.Errorf("a sub-threshold blip opened %d utterances", len(h.utts))
	}
}

// Two separate sub-threshold blips must not accumulate into a barge-in.
func TestBargeInCounterResetsBetweenBlips(t *testing.T) {
	h := newHarness(ArmOnWake, testCfg())
	h.m.SetSpeaking(true)

	for i := 0; i < 3; i++ {
		h.vadStart()
		h.frames(4)
		h.vadEnd()
		h.frames(2)
	}
	if h.barges != 0 {
		t.Errorf("barges = %d, want 0 — blips must not accumulate", h.barges)
	}
}

// The interrupting words must survive: the utterance starts at the frame where
// speech began, not at the frame the threshold was reached.
func TestBargeInKeepsOnsetViaPreroll(t *testing.T) {
	h := newHarness(ArmOnWake, testCfg())
	h.m.SetSpeaking(true)

	h.frames(20) // assistant talking, user silent
	onset := h.seq
	h.vadStart()
	h.frames(6) // triggers the barge-in on the 6th

	h.frames(20)
	h.vadEnd()
	h.frames(15)

	if len(h.utts) != 1 {
		t.Fatalf("want 1 utterance, got %d", len(h.utts))
	}
	u := h.utts[0]
	if u.StartSeq > onset {
		t.Errorf("utterance starts at seq %d, after speech onset at %d — the first words were lost", u.StartSeq, onset)
	}
	// The onset frame's stamped PCM must actually be present in the buffer.
	want := byte(onset)
	found := false
	for i := 0; i < len(u.PCM); i += frameBytes {
		if u.PCM[i] == want {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("frame %d (stamp %d) missing from the utterance buffer", onset, want)
	}
}

// The ASR stream worker is armed with the preroll so the barge-in onset is
// transcribed by the streaming path too, and with the onset's seq — not the
// previous utterance's.
func TestBargeInHandsPrerollToArm(t *testing.T) {
	h := newHarness(ArmOnWake, testCfg())
	h.m.SetSpeaking(true)

	h.frames(20) // assistant talking, user silent
	onset := h.seq
	h.vadStart()
	h.frames(6) // triggers the barge-in on the 6th

	if len(h.arms) != 1 {
		t.Fatalf("want 1 arm on barge-in, got %d", len(h.arms))
	}
	a := h.arms[0]
	if a.startSeq > onset {
		t.Errorf("arm startSeq = %d, after speech onset at %d — the first words were lost", a.startSeq, onset)
	}
	// The ring pads bargeRingSlack frames beyond the confirmation window, so
	// the preroll is at least BargeInFrames long.
	if got := len(a.preroll); got < 6*frameBytes {
		t.Errorf("preroll len = %d, want at least %d (BargeInFrames)", got, 6*frameBytes)
	}
	// The onset frame's stamped PCM must be inside the preroll.
	want := byte(onset)
	found := false
	for i := 0; i < len(a.preroll); i += frameBytes {
		if a.preroll[i] == want {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("frame %d (stamp %d) missing from the preroll", onset, want)
	}
}

// Normal (non-barge) arming must carry no preroll, so the stream worker arms
// the plain way.
func TestNormalArmHasNoPreroll(t *testing.T) {
	h := newHarness(ArmOnVad, testCfg())
	h.vadStart()

	if len(h.arms) != 1 {
		t.Fatalf("want 1 arm, got %d", len(h.arms))
	}
	if got := len(h.arms[0].preroll); got != 0 {
		t.Errorf("normal arm carried preroll of %d bytes", got)
	}
}

// A reply that finishes on its own returns to idle without opening anything.
func TestSpeakingEndReturnsToIdle(t *testing.T) {
	h := newHarness(ArmOnWake, testCfg())
	h.m.SetSpeaking(true)
	h.frames(10)
	h.m.SetSpeaking(false)

	// In ArmOnWake mode a bare speech_start from idle must still be ignored.
	h.vadStart()
	h.frames(30)
	h.vadEnd()
	h.frames(15)

	if h.barges != 0 {
		t.Errorf("barges = %d after the reply ended, want 0", h.barges)
	}
	if len(h.utts) != 0 {
		t.Errorf("speech after the reply ended opened %d utterances without a wake word", len(h.utts))
	}
}

// A turn must not start while an utterance is still open.
func TestSpeakingStartIgnoredWhileCapturing(t *testing.T) {
	h := newHarness(ArmOnVad, testCfg())
	h.vadStart()
	h.frames(20)
	h.m.SetSpeaking(true) // must be a no-op while an utterance is open

	h.vadEnd()
	h.frames(15)
	if len(h.utts) != 1 {
		t.Fatalf("want 1 utterance, got %d — OnSpeakingStart disrupted an open capture", len(h.utts))
	}
	if got := h.utts[0].FrameCount; got != 20 {
		t.Errorf("FrameCount = %d, want 20", got)
	}
}
