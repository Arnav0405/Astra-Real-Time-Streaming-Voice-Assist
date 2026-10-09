package asr

import (
	"context"
	"errors"
	"io"
	"sync/atomic"
	"testing"
)

// TestStreamWorkerCompile tests that the StreamWorker compiles correctly
func TestStreamWorkerCompile(t *testing.T) {
	cfg := &StreamWorkerConfig{
		GRPCAddress:   "localhost:50051",
		Model:         "small",
		Language:      "en",
		ChunkFrames:   150,
		OverlapFrames: 50,
		GraceFrames:   2,
	}

	worker := NewStreamWorker("test-stream", cfg, nil)
	_ = worker
}

// TestStreamWorkerStartStop tests basic start/stop without actual connection
func TestStreamWorkerStartStop(t *testing.T) {
	cfg := &StreamWorkerConfig{
		GRPCAddress:   "localhost:50051",
		Model:         "small",
		Language:      "en",
		ChunkFrames:   150,
		OverlapFrames: 50,
		GraceFrames:   2,
	}

	worker := NewStreamWorker("test-stream", cfg, nil)
	
	// Test Arm/Disarm/BargeIn without calling Start (no connection attempt)
	worker.Arm(1)
	worker.Disarm(100)
	worker.BargeIn(101, []byte{0x01, 0x02, 0x03})
	worker.PushFrame(1, []byte{0x01, 0x02})
	
	// Close without start should not panic
	worker.Close()
}

// TestStreamTranscript tests StreamTranscript struct
func TestStreamTranscript(t *testing.T) {
	t.Parallel()
	
	tr := StreamTranscript{
		StreamID:   "s1",
		Text:       "hello",
		IsFinal:    true,
		StartSeq:   1,
		EndSeq:     10,
		FrameCount: 10,
	}
	
	_ = tr
	if !tr.IsFinal {
		t.Error("expected final")
	}
}

// TestStreamWorkerConfig tests StreamWorkerConfig struct
func TestStreamWorkerConfig(t *testing.T) {
	t.Parallel()
	
	cfg := &StreamWorkerConfig{
		GRPCAddress:   "localhost:50051",
		Model:         "small",
		Language:      "en",
		ChunkFrames:   150,
		OverlapFrames: 50,
		GraceFrames:   2,
	}
	
	_ = cfg
	if cfg.ChunkFrames != 150 {
		t.Error("expected 150 chunk frames")
	}
}

func TestStitch(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		running string
		delta   string
		want    string
	}{
		{"empty running", "", "Hello world", "Hello world"},
		{"empty delta", "Hello world", "", "Hello world"},
		{"both empty", "", "", ""},
		{"no overlap", "abc def", "xyz", "abc def xyz"},
		{"one word overlap", "Hello world", "world today", "Hello world today"},
		{"real whisper overlap", "Check if this is working or not.", "not. That's it.", "Check if this is working or not. That's it."},
		{"delta fully contained", "or not.", "not.", "or not."},
		{"case insensitive overlap", "Or Not.", "not. yes", "Or Not. yes"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := stitch(tc.running, tc.delta); got != tc.want {
				t.Errorf("stitch(%q, %q) = %q, want %q", tc.running, tc.delta, got, tc.want)
			}
		})
	}
}

type scriptedStream struct {
	responses []struct {
		text  string
		final bool
	}
	i      int
	pushed [][]byte
	closed atomic.Bool
}

func (s *scriptedStream) PushPCM(pcm []byte) error {
	s.pushed = append(s.pushed, append([]byte(nil), pcm...))
	return nil
}

func (s *scriptedStream) Recv() (string, bool, error) {
	if s.i >= len(s.responses) {
		return "", false, io.EOF
	}
	r := s.responses[s.i]
	s.i++
	return r.text, r.final, nil
}

func (s *scriptedStream) Close() error {
	s.closed.Store(true)
	return nil
}

type fakeGRPCClient struct {
	utteranceIDs []uint64
	streams      []*scriptedStream
}

func (f *fakeGRPCClient) NewStream(_ context.Context, id uint64) (Stream, error) {
	s := &scriptedStream{}
	f.utteranceIDs = append(f.utteranceIDs, id)
	f.streams = append(f.streams, s)
	return s, nil
}

func (f *fakeGRPCClient) CloseClient() error { return nil }

// The barge-in preroll must reach the ASR service as a single PCM chunk, not
// one gRPC message per byte.
func TestBargeInPushesPrerollAsOneChunk(t *testing.T) {
	t.Parallel()

	f := &fakeGRPCClient{}
	w := NewStreamWorker("test-stream", &StreamWorkerConfig{ChunkFrames: 150}, nil)
	w.grpcClient = f

	preroll := make([]byte, 3*640)
	w.handleControl(workerMsg{kind: msgBarge, seq: 100, pcm: preroll})

	if len(f.utteranceIDs) != 1 || f.utteranceIDs[0] != 1 {
		t.Fatalf("utteranceIDs = %v, want [1]", f.utteranceIDs)
	}
	stream := f.streams[0]
	if len(stream.pushed) != 1 {
		t.Fatalf("preroll sent as %d gRPC messages, want 1", len(stream.pushed))
	}
	if len(stream.pushed[0]) != len(preroll) {
		t.Errorf("pushed chunk len = %d, want %d", len(stream.pushed[0]), len(preroll))
	}
}

// After a barge-in, the worker must be armed: live frames flow into the new
// stream and Disarm half-closes it, which is what makes Whisper finalize and
// the final transcript (the LLM's input) appear.
func TestBargeInArmsWorkerForNewFrames(t *testing.T) {
	t.Parallel()

	f := &fakeGRPCClient{}
	w := NewStreamWorker("test-stream", &StreamWorkerConfig{ChunkFrames: 150}, nil)
	w.grpcClient = f

	w.handleControl(workerMsg{kind: msgBarge, seq: 100, pcm: make([]byte, 2*640)})

	// Frames arriving after the barge-in must reach the new stream.
	w.handleFrame(workerMsg{kind: msgFrame, seq: 101, pcm: make([]byte, 640)})
	s := f.streams[0]
	if got, want := len(s.pushed), 2; got != want {
		t.Fatalf("barged stream received %d chunks (preroll + live frames), want %d", got, want)
	}
	if got, want := len(s.pushed[1]), 640; got != want {
		t.Errorf("live frame chunk len = %d, want %d", got, want)
	}

	// Closing the utterance must half-close the gRPC stream so the server
	// finalizes and emits the final transcript.
	w.handleControl(workerMsg{kind: msgDisarm})
	if !s.closed.Load() {
		t.Error("Disarm after barge-in did not half-close the gRPC stream; Whisper would never finalize")
	}
}

func TestRecvLoopAccumulatesTranscripts(t *testing.T) {
	t.Parallel()

	var got []StreamTranscript
	w := NewStreamWorker("test-stream", &StreamWorkerConfig{ChunkFrames: 150}, func(st StreamTranscript) {
		got = append(got, st)
	})

	s := &scriptedStream{}
	s.responses = []struct {
		text  string
		final bool
	}{
		{"Hello world", false},
		{"world today", false},
		{"", true},
	}
	w.recvLoop(s, 10)

	want := []struct {
		text  string
		final bool
	}{
		{"Hello world", false},
		{"Hello world today", false},
		{"Hello world today", true},
	}
	if len(got) != len(want) {
		t.Fatalf("recvLoop emitted %d transcripts, want %d: %+v", len(got), len(want), got)
	}
	for i, wnt := range want {
		if got[i].Text != wnt.text || got[i].IsFinal != wnt.final {
			t.Errorf("transcript %d = %q (final=%v), want %q (final=%v)", i, got[i].Text, got[i].IsFinal, wnt.text, wnt.final)
		}
	}
}

func TestRecvLoopStitchesOverlap(t *testing.T) {
	t.Parallel()

	var got []StreamTranscript
	w := NewStreamWorker("test-stream", &StreamWorkerConfig{ChunkFrames: 150}, func(st StreamTranscript) {
		got = append(got, st)
	})

	s := &scriptedStream{}
	s.responses = []struct {
		text  string
		final bool
	}{
		{"Check if this is working or not.", false},
		{"not. That's it. That's all.", false},
		{"That's all. Thank you.", true},
	}
	w.recvLoop(s, 0)

	want := []string{
		"Check if this is working or not.",
		"Check if this is working or not. That's it. That's all.",
		"Check if this is working or not. That's it. That's all. Thank you.",
	}
	if len(got) != len(want) {
		t.Fatalf("recvLoop emitted %d transcripts, want %d: %+v", len(got), len(want), got)
	}
	for i, wnt := range want {
		if got[i].Text != wnt {
			t.Errorf("transcript %d = %q, want %q", i, got[i].Text, wnt)
		}
	}
	if !got[len(got)-1].IsFinal {
		t.Error("last transcript should be final")
	}
}

// Every frame pushed right after Arm must reach the gRPC stream. Arm and the
// frame behind it travel one ordered queue; with two channels behind a select
// the frame won the race about half the time and was discarded by the
// "not armed" check, so utterances lost their opening frames at random.
func TestArmThenFrameNeverDropsTheFrame(t *testing.T) {
	const iters = 200
	for i := 0; i < iters; i++ {
		f := &fakeGRPCClient{}
		w := NewStreamWorker("s", &StreamWorkerConfig{ChunkFrames: 150}, nil)
		w.grpcClient = f
		w.running.Store(true) // what Start() does, without the real client
		go w.run()

		w.Arm(0)
		w.PushFrame(0, make([]byte, 640))
		w.Close() // drains the queue, so both messages are applied

		if len(f.streams) != 1 {
			t.Fatalf("iteration %d: %d ASR streams, want 1", i, len(f.streams))
		}
		if got := len(f.streams[0].pushed); got != 1 {
			t.Fatalf("iteration %d: stream received %d chunks after Arm, want 1", i, got)
		}
	}
}

// Guard (passes before and after; pins crash-safety): an unreachable ASR service
// must leave the worker disarmed and drop frames quietly, not panic or push into
// a nil stream. Needs `errors` added to the test file's imports.
type failingGRPCClient struct{}

func (failingGRPCClient) NewStream(context.Context, uint64) (Stream, error) {
	return nil, errors.New("asr unreachable")
}

func (failingGRPCClient) CloseClient() error { return nil }

func TestArmWithUnreachableASRLeavesWorkerDisarmed(t *testing.T) {
	w := NewStreamWorker("s", &StreamWorkerConfig{ChunkFrames: 150}, nil)
	w.grpcClient = failingGRPCClient{}
	w.running.Store(true)
	go w.run()

	w.Arm(0)
	w.PushFrame(0, make([]byte, 640))
	w.Close() // drains the queue: Arm and the frame are both processed

	if w.armed.Load() {
		t.Error("worker armed without a stream")
	}
}

// A full queue must discard the OLDEST audio and count it, not refuse the newest
// frame silently: ASR falling behind is a fact to log, and the audio the user is
// speaking right now is the audio worth transcribing.
func TestFullQueueDropsOldestAndCounts(t *testing.T) {
	w := NewStreamWorker("s", &StreamWorkerConfig{ChunkFrames: 150}, nil)

	for i := 0; i < queueDepth; i++ {
		if !w.enqueue(workerMsg{kind: msgFrame, seq: uint64(i), pcm: make([]byte, 640)}) {
			t.Fatalf("queue rejected frame %d before it was full", i)
		}
	}
	newest := uint64(queueDepth)
	if !w.enqueue(workerMsg{kind: msgFrame, seq: newest, pcm: make([]byte, 640)}) {
		t.Fatal("enqueue refused the newest frame instead of dropping the oldest")
	}
	if got := w.dropped.Load(); got != 1 {
		t.Errorf("dropped = %d, want 1", got)
	}

	// The queue kept the newest audio: seq 0 is gone, `newest` is in. Read a
	// fixed count — the channel is open, so `for range` would block forever.
	first := <-w.queue
	if first.seq != 1 {
		t.Errorf("head of queue = seq %d, want 1 (seq 0 should have been dropped)", first.seq)
	}
	var last workerMsg
	for i := 0; i < queueDepth-1; i++ {
		last = <-w.queue
	}
	if last.seq != newest {
		t.Errorf("tail of queue = seq %d, want %d", last.seq, newest)
	}
}

// The counters are per worker: one stream falling behind must never be reported
// as another stream's loss.
func TestDropCountsArePerWorker(t *testing.T) {
	busy := NewStreamWorker("busy", &StreamWorkerConfig{ChunkFrames: 150}, nil)
	idle := NewStreamWorker("idle", &StreamWorkerConfig{ChunkFrames: 150}, nil)

	for i := 0; i <= queueDepth; i++ { // one past capacity
		busy.enqueue(workerMsg{kind: msgFrame, seq: uint64(i), pcm: make([]byte, 640)})
	}
	if got := busy.dropped.Load(); got == 0 {
		t.Fatal("busy worker reported no drops after overflowing its queue")
	}
	if got := idle.dropped.Load(); got != 0 {
		t.Errorf("idle worker reported %d drops from another stream's backlog", got)
	}
}
