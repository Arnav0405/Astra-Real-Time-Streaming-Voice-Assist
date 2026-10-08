package tts

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/arnav/astra/services/backend/internal/tts/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const requestTimeout = 30 * time.Second

// Synthesizer turns one span of reply text into streamed PCM. *Client
// implements it; tests substitute a fake.
type Synthesizer interface {
	Speak(ctx context.Context, text string, onPCM func([]byte) error) error
}

// Client synthesizes speech over the astra.v1.TTS gRPC service (local Piper
// server). Safe for concurrent use. No retry, for the same reason as llm: a
// retried sentence arrives after the conversation has moved on.
type Client struct {
	cfg     Config
	conn    *grpc.ClientConn
	timeout time.Duration
}

// NewTTSClient returns a client for cfg. It uses grpc.NewClient (lazy, no
// I/O): connection errors surface at RPC time, not here. See antipatterns.md.
func NewTTSClient(cfg Config) (*Client, error) {
	conn, err := grpc.NewClient(
		cfg.GRPCAddress,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("create TTS client for %s: %w", cfg.GRPCAddress, err)
	}
	return &Client{cfg: cfg, conn: conn, timeout: requestTimeout}, nil
}

// CloseClient closes the underlying gRPC connection. Main defers it at
// shutdown, mirroring the ASR client.
func (c *Client) CloseClient() error {
	return c.conn.Close()
}

// SampleRateHz is the rate of the PCM Speak produces.
func (c *Client) SampleRateHz() int { return c.cfg.SampleRateHz }

// Speak synthesizes text and hands PCM to onPCM in arrival order, from the
// calling goroutine. The buffer passed to onPCM comes straight off the wire
// and is not reused by this client, so a consumer that keeps it must copy.
// Returns early if onPCM errors (the consumer is gone) or ctx is cancelled
// (barge-in); cancellation also stops the server mid-stream, since the py
// server checks context.is_active() between chunks.
func (c *Client) Speak(ctx context.Context, text string, onPCM func([]byte) error) error {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Raw-stream form, as in asr: the generated pb package carries only the
	// message types, not the service stub (go_package points into internal).
	desc := &grpc.StreamDesc{
		StreamName:    "Synthesize",
		ServerStreams: true,
	}
	stream, err := grpc.NewClientStream(ctx, desc, c.conn, "/astra.v1.TTS/Synthesize")
	if err != nil {
		return fmt.Errorf("tts create stream: %w", err)
	}
	if err := stream.SendMsg(&pb.SynthesizeRequest{Text: text}); err != nil {
		stream.CloseSend()
		return fmt.Errorf("tts send request: %w", err)
	}
	if err := stream.CloseSend(); err != nil {
		return fmt.Errorf("tts close send: %w", err)
	}

	for {
		var resp pb.SynthesizeResponse
		if err := stream.RecvMsg(&resp); err != nil {
			if errors.Is(err, io.EOF) {
				return nil // server half-closed after the last chunk
			}
			if ctxErr := ctx.Err(); ctxErr != nil {
				return ctxErr // barge-in cancellation, not a TTS failure
			}
			return fmt.Errorf("tts stream: %w", err)
		}
		switch msg := resp.Msg.(type) {
		case *pb.SynthesizeResponse_AudioStart:
			// sample_rate_hz rides AudioStart; the caller already wired the
			// declared cfg rate, so this stays informational.
			_ = msg.AudioStart.GetSampleRateHz()
		case *pb.SynthesizeResponse_AudioChunk:
			if err := onPCM(msg.AudioChunk.GetPcm()); err != nil {
				return err // consumer gone: stop, drop the rest
			}
		}
	}
}
