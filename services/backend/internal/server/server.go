// Package server implements the Phase 1 streaming audio ingest: a WebSocket
// endpoint that accepts protobuf-framed ClientMessages, validates the Frame
// protocol strictly, and drains accepted Frames into a per-session sink.
package server

import (
	"log"
	"net/http"
	"strings"

	"github.com/coder/websocket"

	"github.com/arnav/astra/services/backend/internal/pb"
)

// Frame is one validated 20 ms chunk of PCM audio from a Stream.
type Frame struct {
	Seq uint64
	PCM []byte
}

// Sender writes one ServerMessage to the client. It is safe for concurrent
// use and returns an error once the session is tearing down, so a reply in
// flight when the client disconnects fails instead of blocking. Sinks that
// talk back to the client (Phase 7) hold one.
type Sender func(*pb.ServerMessage) error

// Server upgrades HTTP requests to WebSocket sessions. NewSink is the seam
// downstream stages plug into; it is called once per session, with the
// session's Sender.
type Server struct {
	NewSink func(streamID string, send Sender) Sink
}

// New returns a Server wired to the default stats sink.
func New() *Server {
	return &Server{NewSink: func(streamID string, _ Sender) Sink { return newStatsSink(streamID) }}
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Plain HTTP requests landing on this route (Chrome prerenders bare
	// "http://host/" from history on every reload) would otherwise fail the
	// upgrade with a logged protocol violation. Send them to the web client.
	if !strings.Contains(r.Header.Get("Connection"), "Upgrade") {
		http.Redirect(w, r, "/app/", http.StatusFound)
		return
	}
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
