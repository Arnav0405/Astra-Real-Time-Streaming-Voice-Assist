package asr

import (
	"testing"
	"time"
)

// NewGRPCClient must use lazy grpc.NewClient semantics: it must return
// quickly even when the server is unreachable, and surface connection
// problems at RPC time, not dial time (see antipatterns.md).
// The old grpc.DialContext + WithBlock implementation blocks forever here
// because it uses context.Background().
func TestNewGRPCClientDoesNotBlockWhenServerUnreachable(t *testing.T) {
	cfg := &Config{
		GRPCAddress: "127.0.0.1:1", // closed port: nothing listening
		Model:       "small",
		Language:    "en",
	}

	type result struct {
		client GRPCClient
		err    error
	}
	done := make(chan result, 1)
	go func() {
		c, err := NewGRPCClient(cfg)
		done <- result{client: c, err: err}
	}()

	select {
	case r := <-done:
		if r.err != nil {
			t.Fatalf("NewGRPCClient should succeed lazily, got error: %v", r.err)
		}
		if r.client == nil {
			t.Fatal("NewGRPCClient returned nil client")
		}
		_ = r.client.CloseClient()
	case <-time.After(3 * time.Second):
		t.Fatal("NewGRPCClient blocked >3s on unreachable server: uses DialContext+WithBlock anti-pattern, want lazy grpc.NewClient")
	}
}
