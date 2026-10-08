package tts

import (
	"bytes"
	"context"
	"errors"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/arnav/astra/services/backend/internal/tts/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

func testConfig(addr string) Config {
	return Config{
		GRPCAddress:  addr,
		Voice:        "en_US-lessac-medium",
		SampleRateHz: 22050,
		LengthScale:  1.0, NoiseScale: 0.667, NoiseW: 0.333,
	}
}

// fakeServer implements astra.v1.TTS/Synthesize the same way the py Piper
// server does: one AudioStart, then the body in whole-sample-aligned chunks.
type fakeServer struct {
	body  []byte
	flush int
}

func (s *fakeServer) Synthesize(req *pb.SynthesizeRequest, stream pb.TTS_SynthesizeServer) error {
	if strings.TrimSpace(req.GetText()) == "" {
		return errors.New("empty text")
	}
	if err := stream.Send(&pb.SynthesizeResponse{
		Msg: &pb.SynthesizeResponse_AudioStart{AudioStart: &pb.AudioStart{SampleRateHz: 22050}},
	}); err != nil {
		return err
	}
	for i := 0; i < len(s.body); i += s.flush {
		end := min(i+s.flush, len(s.body))
		chunk := s.body[i:end]
		if len(chunk)%2 == 1 {
			// py server parity: chunks are always whole s16le samples, so an
			// odd tail is carried into the next (1-byte) message.
			if i+s.flush >= len(s.body) {
				chunk = chunk[:len(chunk)-1] // lone trailing byte: drop, like py
			} else {
				// merge into the next message by re-slicing the loop
				next := min(i+s.flush*2, len(s.body))
				chunk = s.body[i:next]
				i += s.flush
			}
		}
		if err := stream.Send(&pb.SynthesizeResponse{
			Msg: &pb.SynthesizeResponse_AudioChunk{
				AudioChunk: &pb.TtsAudioChunk{Pcm: chunk, Seq: uint64(i / s.flush)},
			},
		}); err != nil {
			return err
		}
	}
	return nil
}

// startFake serves the fake over a bufconn and returns a client pointed at
// it, built with the same lazy grpc.NewClient construction NewTTSClient
// uses; only the dialer is swapped for the bufconn. The returned stop func
// shuts the server down.
func startFake(t *testing.T, body []byte, flush int) (*Client, func()) {
	t.Helper()
	srv := grpc.NewServer()
	pb.RegisterTTSServer(srv, &fakeServer{body: body, flush: flush})
	lis := bufconn.Listen(1 << 20)
	go func() { _ = srv.Serve(lis) }()

	conn, err := grpc.NewClient("passthrough:///bufconn",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("grpc client: %v", err)
	}
	return &Client{cfg: testConfig("passthrough:///bufconn"), conn: conn, timeout: requestTimeout}, srv.Stop
}

// serveBody runs Speak against a server that sends body in flush-sized
// messages, which exercises the seams between chunks.
func serveBody(t *testing.T, body []byte, flush int) ([]byte, error) {
	t.Helper()
	client, stop := startFake(t, body, flush)
	defer stop()

	var got bytes.Buffer
	err := client.Speak(context.Background(), "hello", func(p []byte) error {
		if len(p)%2 != 0 {
			t.Errorf("chunk of %d bytes splits an s16le sample", len(p))
		}
		got.Write(p) // Write copies, so wire-buffer reuse is fine
		return nil
	})
	return got.Bytes(), err
}

// TestSpeakPreservesBytesAcrossChunks is the alignment guard, migrated from
// the HTTP tests: what the server sends must reach onPCM in order and whole,
// with no byte dropped or duplicated at message seams.
func TestSpeakPreservesBytesAcrossChunks(t *testing.T) {
	body := make([]byte, 10000)
	for i := range body {
		body[i] = byte(i % 251)
	}
	for _, chunk := range []int{2, 4, 8192} {
		got, err := serveBody(t, body, chunk)
		if err != nil {
			t.Fatalf("chunk=%d: Speak: %v", chunk, err)
		}
		if !bytes.Equal(got, body) {
			t.Errorf("chunk=%d: got %d bytes, want %d, equal=%v", chunk, len(got), len(body), bytes.Equal(got, body))
		}
	}
}

func TestSpeakStopsWhenConsumerErrors(t *testing.T) {
	body := make([]byte, 100000)
	client, stop := startFake(t, body, 2048)
	defer stop()

	sentinel := errors.New("client gone")
	calls := 0
	err := client.Speak(context.Background(), "hi", func([]byte) error {
		calls++
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("want sentinel error, got %v", err)
	}
	if calls != 1 {
		t.Errorf("consumer called %d times after erroring, want 1", calls)
	}
}

func TestSpeakEmptyTextErrors(t *testing.T) {
	client, stop := startFake(t, []byte{0, 0}, 2)
	defer stop()
	err := client.Speak(context.Background(), "   ", func([]byte) error { return nil })
	if err == nil {
		t.Fatal("empty text: want error, got nil")
	}
}

// TestNewTTSClientDoesNotBlockWhenServerUnreachable mirrors the ASR guard:
// the constructor must use lazy grpc.NewClient semantics, surfacing
// connection problems at RPC time, not dial time (see antipatterns.md).
func TestNewTTSClientDoesNotBlockWhenServerUnreachable(t *testing.T) {
	cfg := testConfig("127.0.0.1:1") // closed port: nothing listening
	done := make(chan error, 1)
	go func() {
		c, err := NewTTSClient(cfg)
		if err == nil {
			err = c.CloseClient()
		}
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("NewTTSClient should succeed lazily, got error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("NewTTSClient blocked >3s on unreachable server: uses DialContext+WithBlock anti-pattern, want lazy grpc.NewClient")
	}
}
