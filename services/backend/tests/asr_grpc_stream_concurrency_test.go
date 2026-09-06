package tests

import (
	"context"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/test/bufconn"

	"github.com/arnav/astra/services/backend/internal/asr"
	astrapb "github.com/arnav/astra/services/backend/internal/asr/pb"
)

type testASRService interface {
	StreamTranscribe(grpc.ServerStream) error
}

type testASRServicer struct{}

func (testASRServicer) StreamTranscribe(stream grpc.ServerStream) error {
	chunks := 0
	for {
		var req astrapb.TranscribeRequest
		if err := stream.RecvMsg(&req); err != nil {
			_ = stream.SendMsg(&astrapb.TranscribeResponse{Text: "final", IsFinal: true})
			return nil
		}
		if req.GetChunk() != nil {
			chunks++
			if chunks == 3 {
				if err := stream.SendMsg(&astrapb.TranscribeResponse{Text: "partial", IsFinal: false}); err != nil {
					return err
				}
			}
		}
	}
}

var testASRDesc = grpc.ServiceDesc{
	ServiceName: "astra.v1.ASR",
	HandlerType: (*testASRService)(nil),
	Methods:     []grpc.MethodDesc{},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "StreamTranscribe",
			Handler:       func(srv interface{}, stream grpc.ServerStream) error { return srv.(testASRService).StreamTranscribe(stream) },
			ServerStreams: true,
			ClientStreams: true,
		},
	},
}

func newBufconnClient(t *testing.T) *asr.grpcClient {
	t.Helper()
	lis := bufconn.Listen(1024 * 1024)
	srv := grpc.NewServer()
	srv.RegisterService(&testASRDesc, testASRServicer{})
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.Dial("bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
		grpc.WithInsecure(),
	)
	if err != nil {
		t.Fatalf("dial bufnet: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return &asr.grpcClient{conn: conn, config: &asr.Config{Model: "small", Language: "en"}}
}

func pushWithTimeout(t *testing.T, s asr.Stream, pcm []byte) {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- s.PushPCM(pcm) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("PushPCM: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("PushPCM blocked while Recv was pending: send and recv share a lock (deadlock)")
	}
}

func TestGRPCStreamPushWhileRecvPending(t *testing.T) {
	c := newBufconnClient(t)
	s, err := c.NewStream(context.Background(), 1)
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}

	recvDone := make(chan error, 1)
	go func() {
		text, _, err := s.Recv()
		if err == nil && text != "partial" {
			t.Errorf("expected partial transcript, got %q", text)
		}
		recvDone <- err
	}()

	time.Sleep(100 * time.Millisecond)

	pushWithTimeout(t, s, make([]byte, 640))
	pushWithTimeout(t, s, make([]byte, 640))
	pushWithTimeout(t, s, make([]byte, 640))

	select {
	case err := <-recvDone:
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no response received after 3 chunks")
	}

	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}
