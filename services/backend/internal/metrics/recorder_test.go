package metrics

import (
	"errors"
	"testing"
	"time"

	"github.com/arnav/astra/services/backend/internal/pb"
)

// collect wraps a Recorder's Sender and keeps every message that reached the
// socket, in order — including the Turn messages the recorder injects itself.
func collect(r *Recorder) (send func(*pb.ServerMessage) error, got *[]*pb.ServerMessage) {
	var msgs []*pb.ServerMessage
	inner := func(m *pb.ServerMessage) error {
		msgs = append(msgs, m)
		return nil
	}
	return r.Wrap(inner), &msgs
}

func turns(msgs []*pb.ServerMessage) []*pb.Turn {
	var out []*pb.Turn
	for _, m := range msgs {
		if t := m.GetTurn(); t != nil {
			out = append(out, t)
		}
	}
	return out
}

func spanNames(t *pb.Turn) []string {
	names := make([]string, 0, len(t.GetSpans()))
	for _, s := range t.GetSpans() {
		names = append(names, s.GetName())
	}
	return names
}

func eq(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func transcript(uid uint64) *pb.ServerMessage {
	return &pb.ServerMessage{Msg: &pb.ServerMessage_Transcript{Transcript: &pb.Transcript{UtteranceId: uid}}}
}
func delta(uid uint64) *pb.ServerMessage {
	return &pb.ServerMessage{Msg: &pb.ServerMessage_ReplyDelta{ReplyDelta: &pb.ReplyDelta{UtteranceId: uid}}}
}
func audio(uid uint64) *pb.ServerMessage {
	return &pb.ServerMessage{Msg: &pb.ServerMessage_ReplyAudio{ReplyAudio: &pb.ReplyAudio{UtteranceId: uid}}}
}
func cancel(uid uint64) *pb.ServerMessage {
	return &pb.ServerMessage{Msg: &pb.ServerMessage_Cancel{Cancel: &pb.Cancel{UtteranceId: uid}}}
}
func replyEnd(uid uint64, reason pb.ReplyEnd_Reason) *pb.ServerMessage {
	return &pb.ServerMessage{Msg: &pb.ServerMessage_ReplyEnd{ReplyEnd: &pb.ReplyEnd{UtteranceId: uid, Reason: reason}}}
}

// runTurn drives one whole wake-mode turn through a recorder. Frames are fed
// in step with the events, as the sink does: an event's retroactive index
// always points at a frame that has already arrived, never at a future one.
func runTurn(r *Recorder, send func(*pb.ServerMessage) error, uid uint64) {
	for seq := uint64(0); seq <= 2; seq++ {
		r.Frame(seq)
	}
	r.Arm(2)
	for seq := uint64(3); seq <= 6; seq++ {
		r.Frame(seq)
	}
	r.VadEnd(6)
	r.Utterance(uid)
	_ = send(transcript(uid))
	_ = send(delta(uid))
	_ = send(audio(uid))
}

func TestTurnChainSpans(t *testing.T) {
	r := New("s1", true)
	send, got := collect(r)
	runTurn(r, send, 42)

	all := turns(*got)
	if len(all) != 1 {
		t.Fatalf("want 1 Turn, got %d", len(all))
	}
	turn := all[0]
	want := []string{"wake_detect", "user_speech", "endpoint_tail", "asr", "llm_ttft", "tts_ttfb"}
	if !eq(spanNames(turn), want) {
		t.Fatalf("spans = %v, want %v", spanNames(turn), want)
	}
	if turn.GetUtteranceId() != 42 || turn.GetChain() != "turn" {
		t.Fatalf("uid/chain = %d/%q", turn.GetUtteranceId(), turn.GetChain())
	}
	var last uint32
	for _, s := range turn.GetSpans() {
		if s.GetStartMs() < last {
			t.Fatalf("span %s starts at %d, before previous %d", s.GetName(), s.GetStartMs(), last)
		}
		last = s.GetStartMs()
	}
	// The headline is the machine time only: it must not include how long the
	// user spoke, nor the detector's own lag.
	if turn.GetHeadlineMs() != 0 {
		t.Fatalf("headline = %d, want 0 for an instant fake turn", turn.GetHeadlineMs())
	}
}

func TestTurnEmittedAfterFirstAudioOnly(t *testing.T) {
	r := New("s1", true)
	send, got := collect(r)
	runTurn(r, send, 7)
	_ = send(audio(7))
	_ = send(audio(7))

	if n := len(turns(*got)); n != 1 {
		t.Fatalf("want exactly 1 Turn across 3 ReplyAudio, got %d", n)
	}
	// It must land immediately after the first ReplyAudio, not at the end.
	idx := -1
	for i, m := range *got {
		if m.GetTurn() != nil {
			idx = i
			break
		}
	}
	if idx != 3 {
		t.Fatalf("Turn at index %d, want 3 (transcript, delta, audio, turn)", idx)
	}
}

func TestBargeChain(t *testing.T) {
	r := New("s1", true)
	send, got := collect(r)
	runTurn(r, send, 5)

	for seq := uint64(10); seq < 20; seq++ {
		r.Frame(seq)
	}
	r.VadStart(12) // the interrupting speech began here
	r.Barge()      // confirmed six frames later
	_ = send(cancel(5))

	all := turns(*got)
	if len(all) != 2 {
		t.Fatalf("want turn + barge, got %d", len(all))
	}
	b := all[1]
	if b.GetChain() != "barge" || b.GetUtteranceId() != 5 {
		t.Fatalf("barge chain = %q uid %d", b.GetChain(), b.GetUtteranceId())
	}
	if want := []string{"barge_detect", "cancel_send"}; !eq(spanNames(b), want) {
		t.Fatalf("barge spans = %v, want %v", spanNames(b), want)
	}
	if b.GetHeadlineMs() != 0 {
		t.Fatalf("barge headline = %d, want 0", b.GetHeadlineMs())
	}
}

// Audio from the interrupted turn keeps arriving after the barge-in has
// already opened the next utterance. That straggler must not be mistaken for
// the new chain's first audio — doing so reported a turn whose only real span
// was the endpoint tail, attributed to the wrong utterance.
func TestStragglingAudioFromOldTurnIsIgnored(t *testing.T) {
	r := New("s1", true)
	send, got := collect(r)
	runTurn(r, send, 5)

	// The user interrupts: a new utterance opens and closes.
	for seq := uint64(7); seq <= 12; seq++ {
		r.Frame(seq)
	}
	r.Arm(7)
	r.VadEnd(12)
	r.Utterance(99)

	_ = send(audio(5)) // the old turn's audio, still draining

	all := turns(*got)
	if len(all) != 1 {
		t.Fatalf("want only the first turn's chain, got %d: %v", len(all), all)
	}
	if all[0].GetUtteranceId() != 5 {
		t.Fatalf("chain reported for utterance %d, want 5", all[0].GetUtteranceId())
	}

	// The new utterance still reports its own chain when its own audio lands.
	_ = send(transcript(99))
	_ = send(delta(99))
	_ = send(audio(99))
	all = turns(*got)
	if len(all) != 2 || all[1].GetUtteranceId() != 99 {
		t.Fatalf("second chain = %v, want one for utterance 99", all[1:])
	}
}

func TestCancelWithoutBargeEmitsNothing(t *testing.T) {
	r := New("s1", true)
	send, got := collect(r)
	runTurn(r, send, 5)
	_ = send(cancel(5)) // stream teardown, not a user interruption

	if n := len(turns(*got)); n != 1 {
		t.Fatalf("want only the turn chain, got %d Turns", n)
	}
}

// An LLM 5xx kills the turn before any delta or audio, so the success path
// would never report it. The ReplyEnd{ERROR} must still emit a chain — ending
// at the last stage that ran — so the waterfall shows how far the turn got
// and how long transcription took.
func TestFailedTurnEmitsPartialChain(t *testing.T) {
	r := New("s1", true)
	send, got := collect(r)
	for seq := uint64(0); seq <= 2; seq++ {
		r.Frame(seq)
	}
	r.Arm(2)
	for seq := uint64(3); seq <= 6; seq++ {
		r.Frame(seq)
	}
	r.VadEnd(6)
	r.Utterance(300)
	_ = send(transcript(300))
	_ = send(replyEnd(300, pb.ReplyEnd_ERROR))

	all := turns(*got)
	if len(all) != 1 {
		t.Fatalf("want 1 Turn, got %d", len(all))
	}
	want := []string{"wake_detect", "user_speech", "endpoint_tail", "asr"}
	if !eq(spanNames(all[0]), want) {
		t.Fatalf("spans = %v, want %v", spanNames(all[0]), want)
	}
	if all[0].GetUtteranceId() != 300 || all[0].GetChain() != "turn" {
		t.Fatalf("uid/chain = %d/%q", all[0].GetUtteranceId(), all[0].GetChain())
	}

	// A second failure marker must not duplicate the chain.
	_ = send(replyEnd(300, pb.ReplyEnd_ERROR))
	if n := len(turns(*got)); n != 1 {
		t.Fatalf("want still 1 Turn after repeat ReplyEnd, got %d", n)
	}
}

// A turn that fails after audio started has already reported its chain; the
// error end must not emit a second one. Same for barge-ins, whose ReplyEnd
// carries BARGED_IN rather than ERROR.
func TestEndedTurnsDoNotReEmit(t *testing.T) {
	r := New("s1", true)
	send, got := collect(r)
	runTurn(r, send, 5)
	_ = send(replyEnd(5, pb.ReplyEnd_ERROR))
	if n := len(turns(*got)); n != 1 {
		t.Fatalf("want only the original chain, got %d", n)
	}

	send2, got2 := collect(r)
	runTurn(r, send2, 9)
	r.Frame(20)
	r.VadStart(20)
	r.Barge()
	_ = send2(cancel(9))
	_ = send2(replyEnd(9, pb.ReplyEnd_BARGED_IN))
	chains := turns(*got2)
	if len(chains) != 2 || chains[0].GetChain() != "turn" || chains[1].GetChain() != "barge" {
		t.Fatalf("want turn then barge and nothing more, got %v", chains)
	}
}

func TestMissingMarksDropSpans(t *testing.T) {
	r := New("s1", true)
	send, got := collect(r)
	r.Frame(0)
	r.Arm(0)
	// The utterance closed without a VAD end ever being seen (a max-length
	// force-close does exactly this), so both spans that hang off it are
	// unknown and must not be invented.
	r.Utterance(1)
	_ = send(transcript(1))
	_ = send(delta(1))
	_ = send(audio(1))

	all := turns(*got)
	if len(all) != 1 {
		t.Fatalf("want 1 Turn, got %d", len(all))
	}
	if want := []string{"wake_detect", "asr", "llm_ttft", "tts_ttfb"}; !eq(spanNames(all[0]), want) {
		t.Fatalf("spans = %v, want %v", spanNames(all[0]), want)
	}
}

func TestRingResolvesRetroactiveFrame(t *testing.T) {
	r := New("s1", true)
	r.Frame(0)
	early := time.Now()
	time.Sleep(20 * time.Millisecond)
	for seq := uint64(1); seq < 5; seq++ {
		r.Frame(seq)
	}
	// Frame 0 arrived before the sleep, so a mark pointing back at it must
	// carry that older time — this is the detection lag the report exists for.
	r.mu.Lock()
	at := r.at(0)
	r.mu.Unlock()
	if at.After(early) {
		t.Fatalf("frame 0 resolved to %v, want <= %v", at, early)
	}
}

func TestRingWraparoundFallsBackToNow(t *testing.T) {
	r := New("s1", true)
	r.Frame(1)
	old := time.Now()
	time.Sleep(10 * time.Millisecond)
	// Frame 1+ringSize occupies the same slot and evicts it, so frame 1 is no
	// longer resolvable and must not resolve to the wrong frame's time.
	r.Frame(1 + ringSize)
	r.mu.Lock()
	at := r.at(1)
	r.mu.Unlock()
	if at.Before(old) {
		t.Fatalf("evicted frame resolved to %v, want a fallback later than %v", at, old)
	}
}

func TestSendErrorSkipsInjection(t *testing.T) {
	r := New("s1", true)
	boom := errors.New("socket gone")
	send := r.Wrap(func(*pb.ServerMessage) error { return boom })
	r.Frame(0)
	r.Arm(0)
	if err := send(audio(1)); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want %v", err, boom)
	}
}

func TestVadOnlyModeArmsOnSpeech(t *testing.T) {
	r := New("s1", false)
	send, got := collect(r)
	r.Frame(0)
	r.Frame(1)
	r.VadStart(1)
	r.Frame(2)
	r.Frame(3)
	r.VadStart(3) // a grace reopen mid-utterance must not restart the chain
	r.Frame(4)
	r.Frame(5)
	r.Frame(6)
	r.VadEnd(6)
	r.Utterance(1)
	_ = send(transcript(1))
	_ = send(delta(1))
	_ = send(audio(1))

	all := turns(*got)
	if len(all) != 1 {
		t.Fatalf("want 1 Turn, got %d", len(all))
	}
	if got := spanNames(all[0])[0]; got != "vad_detect" {
		t.Fatalf("first span = %q, want vad_detect", got)
	}
}
