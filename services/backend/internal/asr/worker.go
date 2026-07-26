package asr

import (
	"log"

	"github.com/arnav/astra/services/backend/internal/endpoint"
)

// queueSize bounds the per-stream backlog. If it ever fills, the conversation
// is already minutes behind — drop the newest and log loud.
const queueSize = 8

// Transcriber turns one utterance's PCM into text. *Client implements it;
// tests substitute a fake.
type Transcriber interface {
	Transcribe(pcm []byte) (string, error)
}

// Transcript is one transcribed utterance, handed to the downstream consumer
// (Phase 7's LLM seam; today a logger).
type Transcript struct {
	StreamID   string
	Text       string
	StartSeq   uint64
	EndSeq     uint64
	FrameCount int
}

// Worker transcribes one stream's utterances serially, in order, off the
// frame-processing path. Enqueue and Close must be called from the same
// goroutine (the sink drives both, like the endpoint machine).
type Worker struct {
	streamID string
	ch       chan endpoint.Utterance
	done     chan struct{}
	closed   bool
}

// NewWorker starts the per-stream transcription goroutine. A nil onTranscript
// logs the transcript server-side.
func NewWorker(streamID string, tr Transcriber, onTranscript func(Transcript)) *Worker {
	if onTranscript == nil {
		onTranscript = func(t Transcript) {
			log.Printf("stream %s: transcript seq %d-%d: %q", t.StreamID, t.StartSeq, t.EndSeq, t.Text)
		}
	}
	w := &Worker{
		streamID: streamID,
		ch:       make(chan endpoint.Utterance, queueSize),
		done:     make(chan struct{}),
	}
	go func() {
		defer close(w.done)
		for u := range w.ch {
			text, err := tr.Transcribe(u.PCM)
			if err != nil {
				log.Printf("stream %s: asr dropped utterance seq %d-%d: %v", streamID, u.StartSeq, u.EndSeq, err)
				continue
			}
			onTranscript(Transcript{
				StreamID:   u.StreamID,
				Text:       text,
				StartSeq:   u.StartSeq,
				EndSeq:     u.EndSeq,
				FrameCount: u.FrameCount,
			})
		}
	}()
	return w
}

// Enqueue hands an utterance to the worker without blocking. Returns false if
// the queue is full (utterance dropped) or the worker is closed.
func (w *Worker) Enqueue(u endpoint.Utterance) bool {
	if w.closed {
		return false
	}
	select {
	case w.ch <- u:
		return true
	default:
		log.Printf("stream %s: asr queue full, dropping utterance seq %d-%d", w.streamID, u.StartSeq, u.EndSeq)
		return false
	}
}

// Close drains: queued and in-flight utterances finish transcribing, then the
// worker goroutine exits. Blocks until done. Idempotent.
func (w *Worker) Close() {
	if !w.closed {
		w.closed = true
		close(w.ch)
	}
	<-w.done
}
