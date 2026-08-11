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

// outQueue bounds the outbound backlog. Reply audio arrives at ~50 msg/s, so
// this is several seconds of slack; a client slower than that is already gone.
const outQueue = 256

// writeTimeout bounds a single conn.Write. The write loop deliberately does
// not use the session context: teardown writes (the final Error) must still
// reach a client whose context was already cancelled by the fatal watcher.
const writeTimeout = 5 * time.Second

// errSessionClosed is returned by Sender once the session is tearing down.
var errSessionClosed = errors.New("session closed")

type session struct {
	conn    *websocket.Conn
	newSink func(streamID string, send Sender) Sink

	state   sessionState
	nextSeq uint64
	frames  chan Frame
	sink    Sink
	cancel  context.CancelFunc

	// Every conn.Write goes through the write loop — coder/websocket allows
	// only one write in flight, and the sink writes replies concurrently with
	// the read loop. outClosed is closed (out never is) so a blocked sender is
	// released without risking a send on a closed channel.
	out        chan []byte
	outClosed  chan struct{}
	writerDone chan struct{}
}

func newSession(conn *websocket.Conn, newSink func(string, Sender) Sink) *session {
	return &session{
		conn:       conn,
		newSink:    newSink,
		state:      awaitingStart,
		out:        make(chan []byte, outQueue),
		outClosed:  make(chan struct{}),
		writerDone: make(chan struct{}),
	}
}

// send marshals msg and hands it to the write loop, blocking only for
// backpressure. Safe for concurrent use — this is the Sender given to sinks.
func (s *session) send(msg *pb.ServerMessage) error {
	b, err := proto.Marshal(msg)
	if err != nil {
		return err
	}
	select {
	case s.out <- b:
		return nil
	case <-s.outClosed:
		return errSessionClosed
	}
}

// writeLoop is the sole writer of the connection. It drains whatever is
// already queued after outClosed so the last reply (or the Error) still ships.
func (s *session) writeLoop() {
	defer close(s.writerDone)
	for {
		select {
		case b := <-s.out:
			s.write(b)
		case <-s.outClosed:
			for {
				select {
				case b := <-s.out:
					s.write(b)
				default:
					return
				}
			}
		}
	}
}

func (s *session) write(b []byte) {
	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()
	// A failed write means the client is gone; keep draining so no sender
	// blocks forever waiting on a socket that will never accept again.
	_ = s.conn.Write(ctx, websocket.MessageBinary, b)
}

// run drives the session until the client stops, disconnects, violates the
// protocol, or the sink fails. It owns the connection and always closes it.
func (s *session) run(ctx context.Context) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	s.cancel = cancel

	go s.writeLoop()

	err := s.loop(ctx)

	if s.frames != nil {
		close(s.frames)
		s.sink.Wait()
	}

	// The sink has finished, so nothing else will send. Queue the final Error
	// (if any), then flush the write loop before closing the connection.
	status, reason := websocket.StatusNormalClosure, ""
	var perr *protocolError
	switch {
	case errors.As(err, &perr):
		s.sendError(perr)
		status, reason = websocket.StatusPolicyViolation, perr.code
	case s.sink != nil && s.sink.Err() != nil:
		s.sendError(&protocolError{codeInternal, s.sink.Err().Error()})
		status, reason = websocket.StatusInternalError, codeInternal
	}

	close(s.outClosed)
	<-s.writerDone
	s.conn.Close(status, reason)
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
	s.sink = s.newSink(streamID, s.send)
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

	return s.send(&pb.ServerMessage{
		Msg: &pb.ServerMessage_StreamStarted{StreamStarted: &pb.StreamStarted{StreamId: streamID}},
	})
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

func (s *session) sendError(perr *protocolError) {
	_ = s.send(&pb.ServerMessage{
		Msg: &pb.ServerMessage_Error{Error: &pb.Error{Code: perr.code, Message: perr.msg}},
	})
}

func newStreamID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
