package asr

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/arnav/astra/services/backend/internal/endpoint"
)

type funcTranscriber func(pcm []byte) (string, error)

func (f funcTranscriber) Transcribe(pcm []byte) (string, error) { return f(pcm) }

func utt(n int) endpoint.Utterance {
	return endpoint.Utterance{
		StreamID:   "s1",
		StartSeq:   uint64(n * 100),
		EndSeq:     uint64(n*100 + 50),
		FrameCount: 50,
		PCM:        []byte{byte(n)},
	}
}

func TestWorkerTranscribesInOrder(t *testing.T) {
	var got []Transcript
	tr := funcTranscriber(func(pcm []byte) (string, error) {
		return fmt.Sprintf("t%d", pcm[0]), nil
	})
	w := NewWorker("s1", tr, func(tt Transcript) { got = append(got, tt) })

	for n := 1; n <= 3; n++ {
		if !w.Enqueue(utt(n)) {
			t.Fatalf("enqueue %d rejected", n)
		}
	}
	w.Close()

	if len(got) != 3 {
		t.Fatalf("got %d transcripts, want 3", len(got))
	}
	for i, tt := range got {
		want := fmt.Sprintf("t%d", i+1)
		if tt.Text != want {
			t.Errorf("transcript[%d].Text = %q, want %q", i, tt.Text, want)
		}
	}
	// Metadata carried through from the utterance.
	if got[0].StreamID != "s1" || got[0].StartSeq != 100 || got[0].EndSeq != 150 || got[0].FrameCount != 50 {
		t.Errorf("transcript[0] metadata = %+v", got[0])
	}
}

func TestWorkerDropsNewestWhenQueueFull(t *testing.T) {
	started := make(chan struct{})
	gate := make(chan struct{})
	var got []Transcript
	tr := funcTranscriber(func(pcm []byte) (string, error) {
		if pcm[0] == 1 {
			started <- struct{}{}
			<-gate
		}
		return "ok", nil
	})
	w := NewWorker("s1", tr, func(tt Transcript) { got = append(got, tt) })

	if !w.Enqueue(utt(1)) {
		t.Fatal("enqueue 1 rejected")
	}
	<-started // utterance 1 is in-flight, queue is empty

	for n := 2; n <= queueSize+1; n++ {
		if !w.Enqueue(utt(n)) {
			t.Fatalf("enqueue %d rejected, queue should have room", n)
		}
	}
	if w.Enqueue(utt(queueSize + 2)) {
		t.Error("enqueue past capacity accepted, want drop")
	}

	close(gate)
	w.Close()
	if len(got) != queueSize+1 {
		t.Errorf("got %d transcripts, want %d", len(got), queueSize+1)
	}
}

func TestWorkerDrainsQueueOnClose(t *testing.T) {
	var got []Transcript
	tr := funcTranscriber(func(pcm []byte) (string, error) {
		time.Sleep(5 * time.Millisecond) // ensure Close races the queue
		return "ok", nil
	})
	w := NewWorker("s1", tr, func(tt Transcript) { got = append(got, tt) })

	for n := 1; n <= 5; n++ {
		if !w.Enqueue(utt(n)) {
			t.Fatalf("enqueue %d rejected", n)
		}
	}
	w.Close()

	if len(got) != 5 {
		t.Errorf("got %d transcripts after Close, want 5 (drain)", len(got))
	}
}

func TestWorkerSkipsFailedTranscription(t *testing.T) {
	var got []Transcript
	tr := funcTranscriber(func(pcm []byte) (string, error) {
		if pcm[0] == 2 {
			return "", errors.New("api down")
		}
		return fmt.Sprintf("t%d", pcm[0]), nil
	})
	w := NewWorker("s1", tr, func(tt Transcript) { got = append(got, tt) })

	for n := 1; n <= 3; n++ {
		w.Enqueue(utt(n))
	}
	w.Close()

	if len(got) != 2 {
		t.Fatalf("got %d transcripts, want 2 (failure dropped)", len(got))
	}
	if got[0].Text != "t1" || got[1].Text != "t3" {
		t.Errorf("texts = %q, %q; want t1, t3", got[0].Text, got[1].Text)
	}
}

func TestWorkerEnqueueAfterCloseRejected(t *testing.T) {
	tr := funcTranscriber(func(pcm []byte) (string, error) { return "ok", nil })
	w := NewWorker("s1", tr, func(Transcript) {})
	w.Close()
	if w.Enqueue(utt(1)) {
		t.Error("enqueue after Close accepted, want reject")
	}
}
