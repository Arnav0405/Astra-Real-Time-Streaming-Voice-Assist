// Package metrics times the two latency chains Astra is judged by: the turn
// chain (arming trigger -> first reply audio) and the barge-in chain (speech
// onset -> Cancel on the wire).
//
// Every boundary is stamped server-side, so both chains end at a socket write.
// What the browser does afterwards — decode, schedule, play — is on the
// client's clock; measuring it would mean synchronising two clocks to report a
// number the server cannot observe, so it is left unmeasured and said so.
//
// Nothing in the pipeline knows this package exists. A Recorder is driven
// entirely from wrappers in cmd/astra: the stage packages keep their
// signatures, exactly as the -verbose tracing already works.
package metrics

import (
	"log/slog"
	"os"
	"sync"
	"time"

	"github.com/arnav/astra/services/backend/internal/pb"
	"github.com/arnav/astra/services/backend/internal/server"
)

// ringSize bounds how far back a retroactive event index can be resolved. The
// deepest is the VAD's own hangover (min_silence 34 frames), so 64 frames of
// 20 ms is comfortable slack.
const ringSize = 64

type slot struct {
	seq uint64
	at  time.Time
	ok  bool
}

// Recorder collects one stream's span marks. Marks arrive on the frame
// goroutine (VAD/wake/barge/utterance) and on the turn goroutine (outbound
// messages), so every method takes the lock.
type Recorder struct {
	streamID string
	// detectLabel names the arming detector's span: the wake word in wake
	// mode, the VAD's own onset lag in VAD-only mode.
	detectLabel string
	log         *slog.Logger

	mu   sync.Mutex
	ring [ringSize]slot

	// Turn chain.
	open      bool
	uid       uint64
	haveUID   bool
	sentTurn  bool
	armAt     time.Time // arrival of the arming event's frame
	armed     time.Time // when the arming callback actually fired
	speechEnd time.Time // arrival of the closing VAD EventEnd's frame
	uttrClose time.Time
	asrDone   time.Time
	llmFirst  time.Time
	ttsFirst  time.Time

	// Barge chain. lastVadStart is kept for every speech onset because
	// whether an onset becomes a barge-in is only known 120 ms later.
	lastVadStart time.Time
	bargeOnset   time.Time
	bargeConfirm time.Time
}

// New builds a Recorder for one stream. armOnWake selects which detector owns
// the first span; it mirrors the endpoint.Mode the same stream runs in.
func New(streamID string, armOnWake bool) *Recorder {
	label := "vad_detect"
	if armOnWake {
		label = "wake_detect"
	}
	return &Recorder{
		streamID:    streamID,
		detectLabel: label,
		log:         slog.New(slog.NewJSONHandler(os.Stderr, nil)),
	}
}

// Frame records a frame's arrival. Called for every inbound frame, so it does
// one map-free ring store and nothing else.
func (r *Recorder) Frame(seq uint64) {
	r.mu.Lock()
	r.ring[seq%ringSize] = slot{seq: seq, at: time.Now(), ok: true}
	r.mu.Unlock()
}

// at resolves a retroactive frame index to its arrival time. A miss means the
// index refers to the frame being processed right now (the sinks call the
// event callbacks before OnFrame), or one older than the ring — either way
// now is the honest answer to within a frame.
func (r *Recorder) at(frame int) time.Time {
	if frame >= 0 {
		if s := r.ring[uint64(frame)%ringSize]; s.ok && s.seq == uint64(frame) {
			return s.at
		}
	}
	return time.Now()
}

// Arm opens a turn chain: the wake word fired, or in VAD-only mode speech
// onset armed an utterance. frame is the event's retroactive index.
func (r *Recorder) Arm(frame int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.resetTurn()
	r.open = true
	r.armAt = r.at(frame)
	r.armed = time.Now()
}

// VadStart notes a speech onset. Only the onset that later turns out to be a
// barge-in matters, and that is not known for another barge_in_frames.
func (r *Recorder) VadStart(frame int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lastVadStart = r.at(frame)
	// VAD-only mode arms on speech onset, but only from idle. An onset while a
	// chain is open and its utterance has not closed yet is a grace reopen,
	// which continues the chain rather than starting one.
	if r.detectLabel == "vad_detect" && (!r.open || !r.uttrClose.IsZero()) {
		r.open = true
		r.armAt = r.lastVadStart
		r.armed = time.Now()
	}
}

// VadEnd notes the end of speech. The last one before the utterance closes is
// the one the endpoint tail is measured from.
func (r *Recorder) VadEnd(frame int) {
	r.mu.Lock()
	r.speechEnd = r.at(frame)
	r.mu.Unlock()
}

// Utterance binds the open chain to its utterance id — the utterance's first
// frame seq, which is already the correlation key on every reply message.
func (r *Recorder) Utterance(id uint64) {
	r.mu.Lock()
	r.uid, r.haveUID = id, true
	r.uttrClose = time.Now()
	r.mu.Unlock()
}

// Barge promotes the last speech onset to a barge-in. Called when the endpoint
// machine confirms one, which is barge_in_frames after the speech began.
func (r *Recorder) Barge() {
	r.mu.Lock()
	r.bargeOnset = r.lastVadStart
	r.bargeConfirm = time.Now()
	r.mu.Unlock()
}

// Wrap returns a Sender that stamps the outbound boundaries and emits the Turn
// message itself. Wrapping the Sender rather than editing internal/server
// keeps every timing concern in one package and one wiring site.
func (r *Recorder) Wrap(send server.Sender) server.Sender {
	return func(msg *pb.ServerMessage) error {
		if err := send(msg); err != nil {
			return err
		}
		if turn := r.observe(msg); turn != nil {
			return send(turn)
		}
		return nil
	}
}

// observe stamps whatever boundary msg represents and returns a Turn message
// when that boundary completes a chain.
func (r *Recorder) observe(msg *pb.ServerMessage) *pb.ServerMessage {
	r.mu.Lock()
	defer r.mu.Unlock()

	switch m := msg.GetMsg().(type) {
	case *pb.ServerMessage_Transcript:
		if r.mine(m.Transcript.GetUtteranceId()) && r.asrDone.IsZero() {
			r.asrDone = time.Now()
		}
	case *pb.ServerMessage_ReplyDelta:
		if r.mine(m.ReplyDelta.GetUtteranceId()) && r.llmFirst.IsZero() {
			r.llmFirst = time.Now()
		}
	case *pb.ServerMessage_ReplyAudio:
		if !r.mine(m.ReplyAudio.GetUtteranceId()) || !r.ttsFirst.IsZero() || r.sentTurn {
			return nil
		}
		r.ttsFirst = time.Now()
		r.sentTurn = true
		return r.turnChain()
	case *pb.ServerMessage_ReplyEnd:
		// A turn that dies before its first audio — an LLM 5xx, a TTS connect
		// failure — would otherwise never be reported at all, yet the timings
		// up to wherever it died are exactly what diagnosing it needs.
		// relative() drops the stages that never ran, so the chain simply ends
		// at the last completed boundary.
		if m.ReplyEnd.GetReason() != pb.ReplyEnd_ERROR || r.sentTurn ||
			!r.mine(m.ReplyEnd.GetUtteranceId()) {
			return nil
		}
		r.sentTurn = true // nothing further can complete this chain
		return r.turnChain()
	case *pb.ServerMessage_Cancel:
		if r.bargeConfirm.IsZero() {
			return nil // cancelled for some reason other than a barge-in
		}
		return r.bargeChain(m.Cancel.GetUtteranceId(), time.Now())
	}
	return nil
}

// mine reports whether a reply message belongs to the chain currently being
// measured. A turn's audio keeps arriving after the next utterance has already
// opened — during a barge-in, or a quick follow-up — and without this a
// straggling ReplyAudio from the old turn would close the *new* chain, which
// had none of its own marks yet.
func (r *Recorder) mine(uid uint64) bool {
	return r.haveUID && uid == r.uid
}

// span is one (name, start, end) triple before it is made relative to T0.
type span struct {
	name       string
	start, end time.Time
}

// turnChain builds the turn waterfall. Spans whose marks are missing are
// dropped rather than drawn as a bar of made-up length — which is also how a
// failed turn is reported: its chain just ends at the last stage that ran.
func (r *Recorder) turnChain() *pb.ServerMessage {
	raw := []span{
		{r.detectLabel, r.armAt, r.armed},
		{"user_speech", r.armed, r.speechEnd},
		{"endpoint_tail", r.speechEnd, r.uttrClose},
		{"asr", r.uttrClose, r.asrDone},
		{"llm_ttft", r.asrDone, r.llmFirst},
		{"tts_ttfb", r.llmFirst, r.ttsFirst},
	}
	spans, t0 := relative(raw)
	if len(spans) == 0 {
		return nil
	}
	// The headline excludes user_speech: how long the tester talked is not
	// latency. It also excludes the detector, which is reported on its own.
	var headline uint32
	for _, s := range spans {
		switch s.GetName() {
		case "endpoint_tail", "asr", "llm_ttft", "tts_ttfb":
			headline += s.GetDurMs()
		}
	}
	t := &pb.Turn{UtteranceId: r.uid, Chain: "turn", Spans: spans, HeadlineMs: headline}
	r.emitLog(t, t0)
	return &pb.ServerMessage{Msg: &pb.ServerMessage_Turn{Turn: t}}
}

// bargeChain builds the interruption waterfall. Its utterance id is the turn
// that was cancelled, so a client can attach it to the bars it already drew.
func (r *Recorder) bargeChain(uid uint64, cancelSent time.Time) *pb.ServerMessage {
	spans, t0 := relative([]span{
		{"barge_detect", r.bargeOnset, r.bargeConfirm},
		{"cancel_send", r.bargeConfirm, cancelSent},
	})
	r.bargeOnset, r.bargeConfirm = time.Time{}, time.Time{}
	if len(spans) == 0 {
		return nil
	}
	t := &pb.Turn{UtteranceId: uid, Chain: "barge", Spans: spans}
	r.emitLog(t, t0)
	return &pb.ServerMessage{Msg: &pb.ServerMessage_Turn{Turn: t}}
}

// relative converts absolute marks to offsets from the chain's T0, dropping
// spans with a missing or backwards mark.
func relative(raw []span) ([]*pb.Span, time.Time) {
	var t0 time.Time
	for _, s := range raw {
		if !s.start.IsZero() && !s.end.IsZero() && !s.end.Before(s.start) {
			t0 = s.start
			break
		}
	}
	if t0.IsZero() {
		return nil, t0
	}
	out := make([]*pb.Span, 0, len(raw))
	for _, s := range raw {
		if s.start.IsZero() || s.end.IsZero() || s.end.Before(s.start) || s.start.Before(t0) {
			continue
		}
		out = append(out, &pb.Span{
			Name:    s.name,
			StartMs: uint32(s.start.Sub(t0).Milliseconds()),
			DurMs:   uint32(s.end.Sub(s.start).Milliseconds()),
		})
	}
	return out, t0
}

// emitLog writes the JSONL line the aggregation script reads. It carries the
// same numbers as the wire message, so the README table and the browser
// waterfall can never disagree.
func (r *Recorder) emitLog(t *pb.Turn, t0 time.Time) {
	attrs := []any{
		slog.String("ev", "turn"),
		slog.String("stream", r.streamID),
		slog.Uint64("uid", t.GetUtteranceId()),
		slog.String("chain", t.GetChain()),
		slog.Time("t0", t0),
	}
	if t.GetChain() == "turn" {
		attrs = append(attrs, slog.Int("headline_ms", int(t.GetHeadlineMs())))
	}
	spans := make([]any, 0, len(t.GetSpans()))
	for _, s := range t.GetSpans() {
		spans = append(spans, slog.Int(s.GetName(), int(s.GetDurMs())))
	}
	attrs = append(attrs, slog.Group("spans", spans...))
	r.log.Info("latency", attrs...)
}

// resetTurn clears the turn chain so the next one starts clean. The barge
// chain is deliberately untouched: it belongs to the reply being interrupted,
// which is a different chain from the utterance the interruption opens.
func (r *Recorder) resetTurn() {
	r.open = false
	r.haveUID, r.sentTurn = false, false
	r.armAt, r.armed = time.Time{}, time.Time{}
	r.speechEnd, r.uttrClose = time.Time{}, time.Time{}
	r.asrDone, r.llmFirst, r.ttsFirst = time.Time{}, time.Time{}, time.Time{}
}
