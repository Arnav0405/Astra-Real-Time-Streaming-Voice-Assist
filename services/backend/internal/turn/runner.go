// Package turn orchestrates one conversational reply: transcript in, streamed
// LLM text and synthesized audio out, cancellable mid-sentence.
//
// The turn runs entirely off the frame path. Barge is the one method the frame
// path calls, and it never blocks: it cancels the turn's context, which aborts
// the in-flight LLM and TTS requests and unwinds the goroutines on their own
// time.
package turn

import (
	"context"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/arnav/astra/services/backend/internal/asr"
	"github.com/arnav/astra/services/backend/internal/llm"
	"github.com/arnav/astra/services/backend/internal/pb"
	"github.com/arnav/astra/services/backend/internal/tts"
)

// paceLead bounds how far ahead of realtime reply audio may be queued.
//
// Without it the turn would push a whole reply into the outbound queue in a
// fraction of the time it takes to play: a later Cancel would then sit behind
// several seconds of audio the user is about to be told to discard, and the
// barge-in would feel laggy no matter how fast detection was. Capping the lead
// caps that worst case.
const paceLead = 300 * time.Millisecond

// sentenceQueue is the LLM→TTS handoff depth. Synthesis is slower than
// generation, so this only smooths bursts; it is not a buffer to grow.
const sentenceQueue = 8

// state is one in-flight turn. barged separates a user interruption from the
// stream simply ending, which cancel alone cannot distinguish.
type state struct {
	cancel context.CancelFunc
	done   chan struct{}
	barged atomic.Bool
}

// Runner drives replies for one stream, one at a time. Start is called from
// the ASR worker goroutine and Barge from the frame path; both are safe
// together.
type Runner struct {
	streamID   string
	send       func(*pb.ServerMessage) error
	streamer   llm.Streamer
	synth      tts.Synthesizer
	rate       int
	onSpeaking func(bool)

	// Conversation history for context-aware replies
	history      []llm.ChatMessage
	maxHistoryTokens int
	maxHistoryTurns  int

	mu     sync.Mutex
	cur    *state
	closed bool
}

// NewRunner returns a reply runner for one stream. rate is the sample rate of
// the PCM synth produces. onSpeaking, if non-nil, is called with true when the
// first audio of a turn goes out and false when the turn ends — the endpoint
// machine uses it to know when barge-in is possible.
// maxHistoryTokens and maxHistoryTurns control conversation history truncation.
func NewRunner(streamID string, send func(*pb.ServerMessage) error, streamer llm.Streamer, synth tts.Synthesizer, rate int, onSpeaking func(bool), maxHistoryTokens, maxHistoryTurns int) *Runner {
	return &Runner{
		streamID:         streamID,
		send:             send,
		streamer:         streamer,
		synth:            synth,
		rate:             rate,
		onSpeaking:       onSpeaking,
		maxHistoryTokens: maxHistoryTokens,
		maxHistoryTurns:  maxHistoryTurns,
	}
}

// Start begins a reply to t, superseding any turn still in flight. It blocks
// until the previous turn has unwound so two turns can never speak at once.
func (r *Runner) Start(t asr.Transcript) {
	r.mu.Lock()
	prev := r.cur
	closed := r.closed
	r.mu.Unlock()

	if prev != nil {
		prev.cancel()
		<-prev.done
	}
	if closed || strings.TrimSpace(t.Text) == "" {
		return // nothing said, or the stream is going away: no turn to run
	}

	ctx, cancel := context.WithCancel(context.Background())
	st := &state{cancel: cancel, done: make(chan struct{})}

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		cancel()
		return
	}
	r.cur = st
	r.mu.Unlock()

	go func() {
		defer close(st.done)
		defer cancel()
		r.run(ctx, st, t)
	}()
}

// Barge cancels the in-flight turn because the user started talking over it.
// Safe to call when nothing is speaking. Never blocks — it is called from the
// frame path.
func (r *Runner) Barge() {
	r.mu.Lock()
	cur := r.cur
	r.mu.Unlock()
	if cur != nil {
		cur.barged.Store(true)
		cur.cancel()
	}
}

// Close cancels any in-flight turn and waits for it to unwind. Idempotent.
func (r *Runner) Close() {
	r.mu.Lock()
	if r.closed {
		cur := r.cur
		r.mu.Unlock()
		if cur != nil {
			<-cur.done
		}
		return
	}
	r.closed = true
	cur := r.cur
	r.mu.Unlock()

	if cur != nil {
		cur.cancel()
		<-cur.done
	}
}

func (r *Runner) run(ctx context.Context, st *state, t asr.Transcript) {
	id := t.StartSeq

	sentences := make(chan string, sentenceQueue)
	var llmErr error

	// Get a copy of current history for this turn
	r.mu.Lock()
	history := make([]llm.ChatMessage, len(r.history))
	copy(history, r.history)
	r.mu.Unlock()

	// Generation and synthesis are pipelined: the first sentence is being
	// spoken while the rest of the reply is still being generated. Running
	// them in lockstep would add the whole generation time to first audio.
	go func() {
		defer close(sentences)
		var buf string
		var assistantReply strings.Builder
		llmErr = r.streamer.StreamWithHistory(ctx, t.Text, history, func(d string) {
			if err := r.emit(&pb.ServerMessage{Msg: &pb.ServerMessage_ReplyDelta{
				ReplyDelta: &pb.ReplyDelta{UtteranceId: id, Text: d},
			}}); err != nil {
				return
			}
			assistantReply.WriteString(d)
			buf += d
			for {
				chunk, rest, ok := nextChunk(buf)
				if !ok {
					break
				}
				buf = rest
				select {
				case sentences <- chunk:
				case <-ctx.Done():
					return
				}
			}
		})
		if rem := strings.TrimSpace(buf); rem != "" {
			select {
			case sentences <- rem:
			case <-ctx.Done():
			}
		}
		// Store assistant reply in history after completion
		if assistantReply.Len() > 0 {
			r.mu.Lock()
			r.history = append(r.history,
				llm.ChatMessage{Role: "user", Content: t.Text},
				llm.ChatMessage{Role: "assistant", Content: assistantReply.String()},
			)
			r.mu.Unlock()
		}
	}()

	var (
		ttsErr    error
		speaking  bool
		seq       uint64
		pcmBytes  int
		startedAt time.Time
	)
	for s := range sentences {
		err := r.synth.Speak(ctx, s, func(pcm []byte) error {
			if !speaking {
				speaking = true
				startedAt = time.Now()
				r.setSpeaking(true)
			}
			if err := r.emit(&pb.ServerMessage{Msg: &pb.ServerMessage_ReplyAudio{
				ReplyAudio: &pb.ReplyAudio{
					UtteranceId:  id,
					Seq:          seq,
					Pcm:          pcm,
					SampleRateHz: uint32(r.rate),
				},
			}}); err != nil {
				return err
			}
			seq++
			pcmBytes += len(pcm)
			return r.pace(ctx, pcmBytes, startedAt)
		})
		if err != nil {
			ttsErr = err
			break
		}
	}
	for range sentences { //nolint:revive // drain so the generator goroutine can exit
	}

	reason := pb.ReplyEnd_DONE
	switch {
	case st.barged.Load():
		reason = pb.ReplyEnd_BARGED_IN
		// Cancel first: the server stopping sending is not the same as the
		// client stopping playing, and the client-side flush is what the user
		// perceives as the interruption.
		r.emit(&pb.ServerMessage{Msg: &pb.ServerMessage_Cancel{Cancel: &pb.Cancel{UtteranceId: id}}})
	case ctx.Err() != nil:
		// Cancelled by Close (stream ending), not by the user. Nothing to say.
	case llmErr != nil:
		reason = pb.ReplyEnd_ERROR
		log.Printf("stream %s: llm failed on utterance %d: %v", r.streamID, id, llmErr)
	case ttsErr != nil:
		reason = pb.ReplyEnd_ERROR
		log.Printf("stream %s: tts failed on utterance %d: %v", r.streamID, id, ttsErr)
	}

	r.emit(&pb.ServerMessage{Msg: &pb.ServerMessage_ReplyEnd{
		ReplyEnd: &pb.ReplyEnd{UtteranceId: id, Reason: reason},
	}})
	if speaking {
		r.setSpeaking(false)
	}
}

// pace sleeps until the audio already sent is no more than paceLead ahead of
// realtime. Returns ctx.Err() if the turn is cancelled while waiting.
func (r *Runner) pace(ctx context.Context, pcmBytes int, startedAt time.Time) error {
	if r.rate <= 0 {
		return nil
	}
	const bytesPerSample = 2 // s16le mono
	played := time.Duration(pcmBytes) * time.Second / time.Duration(r.rate*bytesPerSample)
	ahead := played - time.Since(startedAt)
	if ahead <= paceLead {
		return nil
	}
	t := time.NewTimer(ahead - paceLead)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (r *Runner) emit(msg *pb.ServerMessage) error {
	return r.send(msg)
}

func (r *Runner) setSpeaking(v bool) {
	if r.onSpeaking != nil {
		r.onSpeaking(v)
	}
}
