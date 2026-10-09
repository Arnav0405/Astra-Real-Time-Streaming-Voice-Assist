package server

import (
	"context"
	"net/http"
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

func (c *captureSink) Fatal() <-chan struct{} { return nil }
func (c *captureSink) Err() error             { return nil }

func dial(t *testing.T) (*websocket.Conn, *captureSink) {
	t.Helper()
	sink := &captureSink{done: make(chan struct{})}
	srv := &Server{NewSink: func(string, Sender) Sink { return sink }}
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

// Plain HTTP on the WS route must redirect to the web client, not fail the
// upgrade with a logged protocol violation (Chrome prerenders bare "/" from
// Plain HTTP on the WS route must redirect to the web client, not fail the
// upgrade with a logged protocol violation (Chrome prerenders bare "/" from
// history on every reload).
func TestPlainHTTPRedirectsToApp(t *testing.T) {
	srv := &Server{NewSink: func(string, Sender) Sink { t.Fatal("sink created for plain HTTP"); return nil }}
	ts := httptest.NewServer(srv)
	defer ts.Close()

	req, err := http.NewRequest(http.MethodGet, ts.URL, nil)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	client := &http.Client{
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusFound {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusFound)
	}
	if loc := resp.Header.Get("Location"); loc != "/app/" {
		t.Fatalf("Location = %q, want /app/", loc)
	}
}

// slowSink models the real sink when inference cannot keep up. perFrame MUST be
// slower than frameDropGrace (the session's send deadline): if the sink frees a
// buffer slot inside the grace window, the session's send succeeds and nothing
// is ever dropped, so the test would not exercise backpressure at all.
type slowSink struct {
	done     chan struct{}
	perFrame time.Duration
	frames   []Frame
}

func (s *slowSink) Run(frames <-chan Frame) {
	defer close(s.done)
	for f := range frames {
		time.Sleep(s.perFrame)
		s.frames = append(s.frames, f)
	}
}

func (s *slowSink) Wait()                  { <-s.done }
func (s *slowSink) Fatal() <-chan struct{} { return nil }
func (s *slowSink) Err() error             { return nil }

// A sink slower than realtime must not stall the read loop: the session skips
// forward, keeps the newest audio, and never reorders or repeats a seq.
func TestSlowSinkSkipsForwardInsteadOfStalling(t *testing.T) {
	const n = 120
	sink := &slowSink{done: make(chan struct{}), perFrame: 30 * time.Millisecond}
	srv := &Server{NewSink: func(string, Sender) Sink { return sink }}
	ts := httptest.NewServer(srv)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(ts.URL, "http"), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

	send(t, conn, validStart())
	recv(t, conn) // StreamStarted

	pcm := make([]byte, frameBytes)
	for seq := uint64(0); seq < n; seq++ {
		send(t, conn, frameMsg(seq, pcm))
	}
	send(t, conn, stopMsg())
	conn.Close(websocket.StatusNormalClosure, "")
	sink.Wait()

	if len(sink.frames) == 0 {
		t.Fatal("sink received nothing")
	}
	if sink.frames[0].Seq != 0 {
		t.Errorf("first delivered seq = %d, want 0 (nothing was queued to drop yet)", sink.frames[0].Seq)
	}
	if got := sink.frames[len(sink.frames)-1].Seq; got != n-1 {
		t.Errorf("last delivered seq = %d, want %d (the newest audio must survive)", got, n-1)
	}
	for i := 1; i < len(sink.frames); i++ {
		if sink.frames[i].Seq <= sink.frames[i-1].Seq {
			t.Fatalf("seq %d followed %d: the session reordered or duplicated frames",
				sink.frames[i].Seq, sink.frames[i-1].Seq)
		}
	}
	if len(sink.frames) == n {
		t.Fatalf("sink kept up with %d frames; the test did not exercise backpressure", n)
	}
}
