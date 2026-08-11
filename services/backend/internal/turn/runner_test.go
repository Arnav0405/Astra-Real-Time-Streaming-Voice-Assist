package turn

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/arnav/astra/services/backend/internal/asr"
	"github.com/arnav/astra/services/backend/internal/pb"
)

const testRate = 24000

// fakeLLM emits deltas, pausing between them so a test can cancel mid-reply.
type fakeLLM struct {
	deltas []string
	gap    time.Duration
	err    error
}

func (f *fakeLLM) Stream(ctx context.Context, _ string, onDelta func(string)) error {
	for _, d := range f.deltas {
		if err := ctx.Err(); err != nil {
			return err
		}
		onDelta(d)
		if f.gap > 0 {
			select {
			case <-time.After(f.gap):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
	return f.err
}

// fakeTTS returns a fixed amount of PCM per sentence and records what it was
// asked to say.
type fakeTTS struct {
	mu     sync.Mutex
	spoken []string
	bytes  int // PCM bytes emitted per sentence
	err    error
}

func (f *fakeTTS) Speak(ctx context.Context, text string, onPCM func([]byte) error) error {
	f.mu.Lock()
	f.spoken = append(f.spoken, text)
	f.mu.Unlock()
	if f.err != nil {
		return f.err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return onPCM(make([]byte, f.bytes))
}

func (f *fakeTTS) said() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.spoken...)
}

// recorder captures the ServerMessages a turn emits. ends lets a test wait for
// a turn to finish on its own — calling Close instead would cancel it, and the
// test would be racing the reply it means to assert on.
type recorder struct {
	mu   sync.Mutex
	msgs []*pb.ServerMessage
	err  error
	ends chan *pb.ReplyEnd
}

func newRecorder() *recorder {
	return &recorder{ends: make(chan *pb.ReplyEnd, 8)}
}

func (r *recorder) send(m *pb.ServerMessage) error {
	r.mu.Lock()
	if r.err != nil {
		err := r.err
		r.mu.Unlock()
		return err
	}
	r.msgs = append(r.msgs, m)
	r.mu.Unlock()

	if e, ok := m.Msg.(*pb.ServerMessage_ReplyEnd); ok {
		select {
		case r.ends <- e.ReplyEnd:
		default:
		}
	}
	return nil
}

// waitEnd blocks until the turn emits its ReplyEnd.
func (r *recorder) waitEnd(t *testing.T) *pb.ReplyEnd {
	t.Helper()
	select {
	case e := <-r.ends:
		return e
	case <-time.After(3 * time.Second):
		t.Fatal("turn never emitted ReplyEnd")
		return nil
	}
}

func (r *recorder) kinds() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []string
	for _, m := range r.msgs {
		switch m.Msg.(type) {
		case *pb.ServerMessage_Transcript:
			out = append(out, "transcript")
		case *pb.ServerMessage_ReplyDelta:
			out = append(out, "delta")
		case *pb.ServerMessage_ReplyAudio:
			out = append(out, "audio")
		case *pb.ServerMessage_Cancel:
			out = append(out, "cancel")
		case *pb.ServerMessage_ReplyEnd:
			out = append(out, "end")
		}
	}
	return out
}

func (r *recorder) end() *pb.ReplyEnd {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, m := range r.msgs {
		if e, ok := m.Msg.(*pb.ServerMessage_ReplyEnd); ok {
			return e.ReplyEnd
		}
	}
	return nil
}

func has(kinds []string, want string) bool {
	for _, k := range kinds {
		if k == want {
			return true
		}
	}
	return false
}

func transcript(text string) asr.Transcript {
	return asr.Transcript{StreamID: "s", Text: text, StartSeq: 42}
}

func TestTurnEmitsTranscriptThenAudioThenEnd(t *testing.T) {
	rec := newRecorder()
	synth := &fakeTTS{bytes: 480}
	r := NewRunner("s", rec.send, &fakeLLM{deltas: []string{"Hello there. ", "How are you? "}}, synth, testRate, nil)
	r.Start(transcript("hi"))
	rec.waitEnd(t)
	r.Close()

	kinds := rec.kinds()
	if len(kinds) == 0 || kinds[0] != "transcript" {
		t.Fatalf("first message = %v, want transcript first", kinds)
	}
	if !has(kinds, "audio") {
		t.Errorf("no audio emitted: %v", kinds)
	}
	if kinds[len(kinds)-1] != "end" {
		t.Errorf("last message = %q, want end", kinds[len(kinds)-1])
	}
	if got := rec.end().Reason; got != pb.ReplyEnd_DONE {
		t.Errorf("reason = %v, want DONE", got)
	}
	if got := rec.end().UtteranceId; got != 42 {
		t.Errorf("utterance_id = %d, want 42 (the utterance's first frame seq)", got)
	}
	// Both sentences reached TTS, split rather than sent as one blob.
	if said := synth.said(); len(said) != 2 {
		t.Errorf("TTS got %d spans, want 2: %q", len(said), said)
	}
}

func TestEmptyTranscriptRunsNoTurn(t *testing.T) {
	rec := newRecorder()
	synth := &fakeTTS{bytes: 480}
	r := NewRunner("s", rec.send, &fakeLLM{deltas: []string{"unused"}}, synth, testRate, nil)
	r.Start(transcript("   "))
	r.Close()

	if kinds := rec.kinds(); len(kinds) != 0 {
		t.Errorf("emitted %v for an empty transcript, want nothing", kinds)
	}
	if said := synth.said(); len(said) != 0 {
		t.Errorf("TTS called with %q for an empty transcript", said)
	}
}

// TestBargeEmitsCancelBeforeEnd is the barge-in contract: Cancel must reach
// the client (so it flushes buffered audio) and the turn must close as
// BARGED_IN rather than DONE.
func TestBargeEmitsCancelBeforeEnd(t *testing.T) {
	rec := newRecorder()
	speaking := make(chan bool, 8)
	synth := &fakeTTS{bytes: 480}
	llm := &fakeLLM{deltas: []string{"One. ", "Two. ", "Three. ", "Four. "}, gap: 50 * time.Millisecond}

	r := NewRunner("s", rec.send, llm, synth, testRate, func(v bool) { speaking <- v })
	r.Start(transcript("hi"))

	select {
	case v := <-speaking:
		if !v {
			t.Fatal("first onSpeaking call was false")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("turn never started speaking")
	}

	r.Barge()
	r.Close()

	kinds := rec.kinds()
	if !has(kinds, "cancel") {
		t.Fatalf("no cancel emitted: %v", kinds)
	}
	ci, ei := -1, -1
	for i, k := range kinds {
		if k == "cancel" && ci < 0 {
			ci = i
		}
		if k == "end" {
			ei = i
		}
	}
	if ci > ei {
		t.Errorf("cancel at %d came after end at %d", ci, ei)
	}
	if got := rec.end().Reason; got != pb.ReplyEnd_BARGED_IN {
		t.Errorf("reason = %v, want BARGED_IN", got)
	}
}

// A turn cancelled by the stream ending is not a barge-in and must not tell
// the client to flush.
func TestCloseWithoutBargeDoesNotCancel(t *testing.T) {
	rec := newRecorder()
	llm := &fakeLLM{deltas: []string{"One. ", "Two. ", "Three. "}, gap: 80 * time.Millisecond}
	r := NewRunner("s", rec.send, llm, &fakeTTS{bytes: 480}, testRate, nil)
	r.Start(transcript("hi"))
	time.Sleep(30 * time.Millisecond)
	r.Close()

	if has(rec.kinds(), "cancel") {
		t.Errorf("Close emitted a cancel: %v", rec.kinds())
	}
}

func TestLLMErrorEndsTurnAsError(t *testing.T) {
	rec := newRecorder()
	r := NewRunner("s", rec.send, &fakeLLM{err: errors.New("boom")}, &fakeTTS{bytes: 480}, testRate, nil)
	r.Start(transcript("hi"))
	rec.waitEnd(t)
	r.Close()

	if got := rec.end().Reason; got != pb.ReplyEnd_ERROR {
		t.Errorf("reason = %v, want ERROR", got)
	}
}

// A new utterance while a reply is in flight supersedes it; the two turns must
// never speak at the same time.
func TestStartSupersedesInFlightTurn(t *testing.T) {
	rec := newRecorder()
	var mu sync.Mutex
	active, maxActive := 0, 0
	synth := &fakeTTS{bytes: 480}
	llm := &fakeLLM{deltas: []string{"One. ", "Two. ", "Three. "}, gap: 40 * time.Millisecond}

	r := NewRunner("s", rec.send, llm, synth, testRate, func(v bool) {
		mu.Lock()
		defer mu.Unlock()
		if v {
			active++
			maxActive = max(maxActive, active)
		} else {
			active--
		}
	})
	r.Start(transcript("first"))
	time.Sleep(60 * time.Millisecond)
	r.Start(transcript("second"))
	r.Close()

	mu.Lock()
	defer mu.Unlock()
	if maxActive > 1 {
		t.Errorf("%d turns spoke concurrently, want at most 1", maxActive)
	}
}

func TestSendFailureAbortsTurn(t *testing.T) {
	rec := newRecorder()
	rec.err = errors.New("client gone")
	synth := &fakeTTS{bytes: 480}
	r := NewRunner("s", rec.send, &fakeLLM{deltas: []string{"Hello there. "}}, synth, testRate, nil)

	done := make(chan struct{})
	go func() { defer close(done); r.Start(transcript("hi")); r.Close() }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("turn did not abort when the client was gone")
	}
	if said := synth.said(); len(said) != 0 {
		t.Errorf("TTS was called after the transcript send failed: %q", said)
	}
}

// pace must hold audio near realtime so a later Cancel is not stuck behind
// seconds of queued playback.
func TestPaceHoldsAudioNearRealtime(t *testing.T) {
	r := NewRunner("s", func(*pb.ServerMessage) error { return nil }, nil, nil, testRate, nil)
	start := time.Now()
	// 2 seconds of 24 kHz s16le audio claimed as already sent.
	twoSeconds := testRate * 2 * 2
	if err := r.pace(context.Background(), twoSeconds, start); err != nil {
		t.Fatalf("pace: %v", err)
	}
	elapsed := time.Since(start)
	want := 2*time.Second - paceLead
	if elapsed < want-100*time.Millisecond {
		t.Errorf("pace returned after %s, want at least ~%s", elapsed, want)
	}
}

func TestPaceReturnsOnCancel(t *testing.T) {
	r := NewRunner("s", func(*pb.ServerMessage) error { return nil }, nil, nil, testRate, nil)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(50 * time.Millisecond); cancel() }()

	start := time.Now()
	err := r.pace(ctx, testRate*2*5, start) // 5 seconds of audio
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("pace err = %v, want context.Canceled", err)
	}
	if time.Since(start) > time.Second {
		t.Errorf("pace ignored cancellation for %s", time.Since(start))
	}
}

func TestChunkSplitting(t *testing.T) {
	tests := []struct {
		name  string
		in    string
		chunk string
		rest  string
		ok    bool
	}{
		{"complete sentence", "Hello there. Next bit", "Hello there. ", "Next bit", true},
		{"no boundary yet", "Hello there", "", "Hello there", false},
		{"decimal not a boundary", "The value is 3.14159 exactly and then some more text", "", "The value is 3.14159 exactly and then some more text", false},
		{"short abbreviation held", "Dr. Smith went home. Then", "Dr. Smith went home. ", "Then", true},
		{"trailing period needs lookahead", "All done.", "", "All done.", false},
		{"newline is a boundary", "First line here\nsecond line", "First line here\n", "second line", true},
		// The min-chunk guard applies to newlines too: a stray early break
		// must not become a two-word TTS call of its own.
		{"newline under min length held", "Short\nline", "", "Short\nline", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			chunk, rest, ok := nextChunk(tc.in)
			if ok != tc.ok || chunk != tc.chunk || rest != tc.rest {
				t.Errorf("nextChunk(%q) = (%q, %q, %v), want (%q, %q, %v)",
					tc.in, chunk, rest, ok, tc.chunk, tc.rest, tc.ok)
			}
		})
	}
}

func TestChunkForceFlushesUnpunctuatedText(t *testing.T) {
	long := strings.Repeat("word ", 100) // no terminal punctuation at all
	chunk, rest, ok := nextChunk(long)
	if !ok {
		t.Fatal("long unpunctuated text never flushed; speech would never start")
	}
	if len(chunk) > maxChunkBytes {
		t.Errorf("chunk of %d bytes exceeds max %d", len(chunk), maxChunkBytes)
	}
	if chunk+rest != long {
		t.Error("force flush lost or duplicated text")
	}
}
