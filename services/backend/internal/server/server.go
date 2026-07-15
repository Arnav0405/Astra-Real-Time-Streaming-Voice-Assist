// Package server implements the Phase 1 streaming audio ingest: a WebSocket
// endpoint that accepts protobuf-framed ClientMessages, validates the Frame
// protocol strictly, and drains accepted Frames into a per-session sink.
package server

import (
	"log"
	"net/http"

	"github.com/coder/websocket"
)

// Frame is one validated 20 ms chunk of PCM audio from a Stream.
type Frame struct {
	Seq uint64
	PCM []byte
}

// Server upgrades HTTP requests to WebSocket sessions. NewSink is the seam
// downstream stages plug into; it is called once per session.
type Server struct {
	NewSink func(streamID string) Sink
}

// New returns a Server wired to the default stats sink.
func New() *Server {
	return &Server{NewSink: func(streamID string) Sink { return newStatsSink(streamID) }}
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	conn, err := websocket.Accept(w, r, nil)
	if err != nil {
		log.Printf("ws accept: %v", err)
		return
	}
	// Largest legal message is ~700 bytes; anything bigger is a violation.
	conn.SetReadLimit(4096)

	sess := newSession(conn, s.NewSink)
	sess.run(r.Context())
}
