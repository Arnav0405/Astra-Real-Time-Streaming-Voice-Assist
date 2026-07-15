package server

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"google.golang.org/protobuf/proto"

	"github.com/arnav/astra/services/backend/internal/pb"
)

// captureSink records everything a session delivers downstream.
type captureSink struct {
	done   chan struct{}
	frames []Frame
}

func (c *captureSink) Run(frames <-chan Frame) {
	defer close(c.done)
	for f := range frames {
		c.frames = append(c.frames, f)
	}
}

func (c *captureSink) Wait() { <-c.done }

func dial(t *testing.T) (*websocket.Conn, *captureSink) {
	t.Helper()
	sink := &captureSink{done: make(chan struct{})}
	srv := &Server{NewSink: func(string) Sink { return sink }}
	ts := httptest.NewServer(srv)
	t.Cleanup(ts.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	conn, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(ts.URL, "http"), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { conn.Close(websocket.StatusNormalClosure, "") })
	return conn, sink
}

func send(t *testing.T, conn *websocket.Conn, msg *pb.ClientMessage) {
	t.Helper()
	data, err := proto.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := conn.Write(ctx, websocket.MessageBinary, data); err != nil {
		t.Fatalf("write: %v", err)
	}
}

func recv(t *testing.T, conn *websocket.Conn) *pb.ServerMessage {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	typ, data, err := conn.Read(ctx)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if typ != websocket.MessageBinary {
		t.Fatalf("expected binary message, got %v", typ)
	}
	var msg pb.ServerMessage
	if err := proto.Unmarshal(data, &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return &msg
}

func validStart() *pb.ClientMessage {
	return &pb.ClientMessage{Msg: &pb.ClientMessage_StreamStart{StreamStart: &pb.StreamStart{
		SampleRateHz:    16000,
		Channels:        1,
		BitsPerSample:   16,
		FrameDurationMs: 20,
	}}}
}

func frameMsg(seq uint64, pcm []byte) *pb.ClientMessage {
	return &pb.ClientMessage{Msg: &pb.ClientMessage_AudioFrame{AudioFrame: &pb.AudioFrame{Seq: seq, Pcm: pcm}}}
}

func stopMsg() *pb.ClientMessage {
	return &pb.ClientMessage{Msg: &pb.ClientMessage_StreamStop{StreamStop: &pb.StreamStop{}}}
}

func TestHappyPath(t *testing.T) {
	conn, sink := dial(t)

	send(t, conn, validStart())
	started := recv(t, conn).GetStreamStarted()
	if started == nil {
		t.Fatal("expected StreamStarted")
	}
	if started.StreamId == "" {
		t.Fatal("expected non-empty stream_id")
	}

	const n = 50
	pcm := make([]byte, frameBytes)
	for i := range pcm {
		pcm[i] = byte(i) // arbitrary non-zero pattern
	}
	for seq := uint64(0); seq < n; seq++ {
		send(t, conn, frameMsg(seq, pcm))
	}
	send(t, conn, stopMsg())
	conn.Close(websocket.StatusNormalClosure, "")
	sink.Wait()

	if len(sink.frames) != n {
		t.Fatalf("sink got %d frames, want %d", len(sink.frames), n)
	}
	for i, f := range sink.frames {
		if f.Seq != uint64(i) {
			t.Fatalf("frame %d has seq %d", i, f.Seq)
		}
		if len(f.PCM) != frameBytes {
			t.Fatalf("frame %d has %d bytes, want %d", i, len(f.PCM), frameBytes)
		}
	}
}

func TestViolations(t *testing.T) {
	pcm := make([]byte, frameBytes)

	cases := []struct {
		name     string
		wantCode string
		drive    func(t *testing.T, conn *websocket.Conn)
	}{
		{
			name:     "frame before start",
			wantCode: codeBadState,
			drive: func(t *testing.T, conn *websocket.Conn) {
				send(t, conn, frameMsg(0, pcm))
			},
		},
		{
			name:     "stop before start",
			wantCode: codeBadState,
			drive: func(t *testing.T, conn *websocket.Conn) {
				send(t, conn, stopMsg())
			},
		},
		{
			name:     "bad format",
			wantCode: codeBadFormat,
			drive: func(t *testing.T, conn *websocket.Conn) {
				msg := validStart()
				msg.GetStreamStart().SampleRateHz = 44100
				send(t, conn, msg)
			},
		},
		{
			name:     "duplicate start",
			wantCode: codeBadState,
			drive: func(t *testing.T, conn *websocket.Conn) {
				send(t, conn, validStart())
				recv(t, conn)
				send(t, conn, validStart())
			},
		},
		{
			name:     "wrong pcm size",
			wantCode: codeBadFrameSize,
			drive: func(t *testing.T, conn *websocket.Conn) {
				send(t, conn, validStart())
				recv(t, conn)
				send(t, conn, frameMsg(0, pcm[:100]))
			},
		},
		{
			name:     "bad seq",
			wantCode: codeBadSeq,
			drive: func(t *testing.T, conn *websocket.Conn) {
				send(t, conn, validStart())
				recv(t, conn)
				send(t, conn, frameMsg(5, pcm))
			},
		},
		{
			name:     "text message",
			wantCode: codeBadMessage,
			drive: func(t *testing.T, conn *websocket.Conn) {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				if err := conn.Write(ctx, websocket.MessageText, []byte("hello")); err != nil {
					t.Fatalf("write: %v", err)
				}
			},
		},
		{
			name:     "garbage bytes",
			wantCode: codeBadMessage,
			drive: func(t *testing.T, conn *websocket.Conn) {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				// Field 2 (audio_frame) wire type 0 — parses as neither valid oneof.
				if err := conn.Write(ctx, websocket.MessageBinary, []byte{0xff, 0xff, 0xff}); err != nil {
					t.Fatalf("write: %v", err)
				}
			},
		},
		{
			name:     "message after stop",
			wantCode: codeBadState,
			drive: func(t *testing.T, conn *websocket.Conn) {
				send(t, conn, validStart())
				recv(t, conn)
				send(t, conn, stopMsg())
				send(t, conn, frameMsg(0, pcm))
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			conn, _ := dial(t)
			tc.drive(t, conn)

			errMsg := recv(t, conn).GetError()
			if errMsg == nil {
				t.Fatal("expected Error message")
			}
			if errMsg.Code != tc.wantCode {
				t.Fatalf("got code %q, want %q", errMsg.Code, tc.wantCode)
			}

			// Connection must be closed by the server after the error.
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if _, _, err := conn.Read(ctx); err == nil {
				t.Fatal("expected connection closed after error")
			}
		})
	}
}
