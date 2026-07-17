package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"time"

	"github.com/coder/websocket"
	"google.golang.org/protobuf/proto"

	"github.com/arnav/astra/services/backend/internal/pb"
)

// Required Frame format — validation targets, not negotiable (decision: strict).
const (
	sampleRateHz    = 16000
	channels        = 1
	bitsPerSample   = 16
	frameDurationMs = 20
	frameBytes      = sampleRateHz / 1000 * frameDurationMs * bitsPerSample / 8 // 640
)

// Violation codes sent in pb.Error before closing.
const (
	codeBadMessage   = "bad_message"    // unparseable or non-binary message
	codeBadState     = "bad_state"      // message illegal in current state
	codeBadFormat    = "bad_format"     // StreamStart fields != required format
	codeBadFrameSize = "bad_frame_size" // pcm length != 640
	codeBadSeq       = "bad_seq"        // seq != previous+1
	codeInternal     = "internal_error" // server-side failure (e.g. VAD inference broke)
)

type sessionState int

const (
	awaitingStart sessionState = iota
	streaming
	stopped
)

// protocolError is a violation the client caused; it is reported back before close.
type protocolError struct {
	code string
	msg  string
}

func (e *protocolError) Error() string { return e.code + ": " + e.msg }

type session struct {
	conn    *websocket.Conn
	newSink func(streamID string) Sink

	state   sessionState
	nextSeq uint64
	frames  chan Frame
	sink    Sink
	cancel  context.CancelFunc
}

func newSession(conn *websocket.Conn, newSink func(string) Sink) *session {
	return &session{conn: conn, newSink: newSink, state: awaitingStart}
}

// run drives the session until the client stops, disconnects, violates the
// protocol, or the sink fails. It owns the connection and always closes it.
func (s *session) run(ctx context.Context) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	s.cancel = cancel

	err := s.loop(ctx)

	if s.frames != nil {
		close(s.frames)
		s.sink.Wait()
	}

	var perr *protocolError
	if errors.As(err, &perr) {
		s.sendError(ctx, perr)
		s.conn.Close(websocket.StatusPolicyViolation, perr.code)
		return
	}
	if s.sink != nil && s.sink.Err() != nil {
		// ctx may already be cancelled by the fatal watcher; use a fresh one
		// so the client still gets the error before close.
		wctx, wcancel := context.WithTimeout(context.Background(), time.Second)
		s.sendError(wctx, &protocolError{codeInternal, s.sink.Err().Error()})
		wcancel()
		s.conn.Close(websocket.StatusInternalError, codeInternal)
		return
	}
	s.conn.Close(websocket.StatusNormalClosure, "")
}

func (s *session) loop(ctx context.Context) error {
	for {
		typ, data, err := s.conn.Read(ctx)
		if err != nil {
			// Client closed (normally or not) or read limit exceeded — for a
			// stopped session that's the expected end, otherwise just over.
			return nil
		}
		if s.state == stopped {
			return &protocolError{codeBadState, "message after StreamStop"}
		}
		if typ != websocket.MessageBinary {
			return &protocolError{codeBadMessage, "expected binary message"}
		}
		var msg pb.ClientMessage
		if err := proto.Unmarshal(data, &msg); err != nil {
			return &protocolError{codeBadMessage, "unparseable ClientMessage"}
		}
		if err := s.handle(ctx, &msg); err != nil {
			return err
		}
	}
}

func (s *session) handle(ctx context.Context, msg *pb.ClientMessage) error {
	switch m := msg.Msg.(type) {
	case *pb.ClientMessage_StreamStart:
		return s.handleStart(ctx, m.StreamStart)
	case *pb.ClientMessage_AudioFrame:
		return s.handleFrame(m.AudioFrame)
	case *pb.ClientMessage_StreamStop:
		if s.state != streaming {
			return &protocolError{codeBadState, "StreamStop before StreamStart"}
		}
		s.state = stopped
		return nil
	default:
		return &protocolError{codeBadMessage, "empty ClientMessage"}
	}
}

func (s *session) handleStart(ctx context.Context, start *pb.StreamStart) error {
	if s.state != awaitingStart {
		return &protocolError{codeBadState, "duplicate StreamStart"}
	}
	if start.SampleRateHz != sampleRateHz || start.Channels != channels ||
		start.BitsPerSample != bitsPerSample || start.FrameDurationMs != frameDurationMs {
		return &protocolError{codeBadFormat, fmt.Sprintf(
			"require %d Hz, %d ch, %d bit, %d ms frames",
			sampleRateHz, channels, bitsPerSample, frameDurationMs)}
	}

	streamID := newStreamID()
	s.state = streaming
	s.frames = make(chan Frame, 32)
	s.sink = s.newSink(streamID)
	go s.sink.Run(s.frames)
	// Abort the read loop if the sink dies mid-stream (a nil Fatal channel,
	// as statsSink returns, never fires).
	go func() {
		select {
		case <-s.sink.Fatal():
			s.cancel()
		case <-ctx.Done():
		}
	}()

	reply, err := proto.Marshal(&pb.ServerMessage{
		Msg: &pb.ServerMessage_StreamStarted{StreamStarted: &pb.StreamStarted{StreamId: streamID}},
	})
	if err != nil {
		return err
	}
	return s.conn.Write(ctx, websocket.MessageBinary, reply)
}

func (s *session) handleFrame(frame *pb.AudioFrame) error {
	if s.state != streaming {
		return &protocolError{codeBadState, "AudioFrame before StreamStart"}
	}
	if len(frame.Pcm) != frameBytes {
		return &protocolError{codeBadFrameSize, fmt.Sprintf("pcm must be %d bytes, got %d", frameBytes, len(frame.Pcm))}
	}
	if frame.Seq != s.nextSeq {
		return &protocolError{codeBadSeq, fmt.Sprintf("expected seq %d, got %d", s.nextSeq, frame.Seq)}
	}
	s.nextSeq++
	s.frames <- Frame{Seq: frame.Seq, PCM: frame.Pcm}
	return nil
}

func (s *session) sendError(ctx context.Context, perr *protocolError) {
	data, err := proto.Marshal(&pb.ServerMessage{
		Msg: &pb.ServerMessage_Error{Error: &pb.Error{Code: perr.code, Message: perr.msg}},
	})
	if err != nil {
		return
	}
	_ = s.conn.Write(ctx, websocket.MessageBinary, data)
}

func newStreamID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
