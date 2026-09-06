package asr

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"google.golang.org/grpc"

	astrapb "github.com/arnav/astra/services/backend/internal/asr/pb"
)

// GRPCClient is the public interface for a gRPC ASR client.
type GRPCClient interface {
	NewStream(ctx context.Context, utteranceID uint64) (Stream, error)
	CloseClient() error
}

// Stream is the bidirectional streaming interface for ASR.
type Stream interface {
	PushPCM(pcm []byte) error
	Recv() (text string, isFinal bool, err error)
	Close() error
}

type grpcClient struct {
	conn   *grpc.ClientConn
	config *Config
}

type grpcStream struct {
	client grpc.ClientStream
	sendMu sync.Mutex
	closed atomic.Bool
}

// NewGRPCClient creates a new gRPC client for the ASR service.
func NewGRPCClient(cfg *Config) (GRPCClient, error) {
	conn, err := grpc.DialContext(
		context.Background(),
		cfg.GRPCAddress,
		grpc.WithInsecure(),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, fmt.Errorf("dial ASR service %s: %w", cfg.GRPCAddress, err)
	}
	return &grpcClient{conn: conn, config: cfg}, nil
}

// NewStream opens a bidirectional streaming RPC for the given utterance ID.
func (c *grpcClient) NewStream(ctx context.Context, utteranceID uint64) (Stream, error) {
	desc := &grpc.StreamDesc{
		StreamName:    "ASR",
		ClientStreams: true,
		ServerStreams: true,
	}

	stream, err := grpc.NewClientStream(ctx, desc, c.conn, "/astra.v1.ASR/StreamTranscribe")
	if err != nil {
		return nil, fmt.Errorf("create ASR stream: %w", err)
	}

	// Send initial config
	configReq := &astrapb.TranscribeRequest{
		Payload: &astrapb.TranscribeRequest_Config{
			Config: &astrapb.StreamConfig{
				UtteranceId:   utteranceID,
				SampleRateHz:  16000,
				Channels:      1,
				BitsPerSample: 16,
				Model:         c.config.Model,
				Language:      c.config.Language,
			},
		},
	}
	if err := stream.SendMsg(configReq); err != nil {
		stream.CloseSend()
		return nil, fmt.Errorf("send config: %w", err)
	}

	return &grpcStream{client: stream}, nil
}

// PushPCM sends a PCM chunk to the streaming ASR service.
func (s *grpcStream) PushPCM(pcm []byte) error {
	if s.closed.Load() {
		return fmt.Errorf("stream already closed")
	}
	s.sendMu.Lock()
	defer s.sendMu.Unlock()
	req := &astrapb.TranscribeRequest{
		Payload: &astrapb.TranscribeRequest_Chunk{
			Chunk: &astrapb.AudioChunk{
				Pcm: pcm,
			},
		},
	}
	return s.client.SendMsg(req)
}

// Recv receives the next transcript response from the streaming ASR service.
func (s *grpcStream) Recv() (text string, isFinal bool, err error) {
	if s.closed.Load() {
		return "", false, fmt.Errorf("stream already closed")
	}
	var resp astrapb.TranscribeResponse
	if err := s.client.RecvMsg(&resp); err != nil {
		return "", false, err
	}
	return resp.GetText(), resp.GetIsFinal(), nil
}

// Close closes the streaming RPC.
func (s *grpcStream) Close() error {
	s.sendMu.Lock()
	defer s.sendMu.Unlock()
	if s.closed.Swap(true) {
		return nil
	}
	return s.client.CloseSend()
}

// CloseClient closes the underlying gRPC connection.
func (c *grpcClient) CloseClient() error {
	return c.conn.Close()
}