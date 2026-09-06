package asr

import (
	"context"
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
	w.handleControl(controlMsg{cmd: 2, seq: 100, pcm: preroll}, new([]byte), new([]uint64), new(uint64), new(int))

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

	w.handleControl(controlMsg{cmd: 2, seq: 100, pcm: make([]byte, 2*640)}, new([]byte), new([]uint64), new(uint64), new(int))

	// Frames arriving after the barge-in must reach the new stream.
	w.handleFrame(frameMsg{seq: 101, pcm: make([]byte, 640)}, new([]byte), new([]uint64), new(uint64), new(int))
	s := f.streams[0]
	if got, want := len(s.pushed), 2; got != want {
		t.Fatalf("barged stream received %d chunks (preroll + live frames), want %d", got, want)
	}
	if got, want := len(s.pushed[1]), 640; got != want {
		t.Errorf("live frame chunk len = %d, want %d", got, want)
	}

	// Closing the utterance must half-close the gRPC stream so the server
	// finalizes and emits the final transcript.
	w.handleControl(controlMsg{cmd: 1}, nil, nil, nil, nil)
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