package tests

import (
	"testing"

	"github.com/arnav/astra/services/backend/internal/asr"
)

// TestGRPCClientCompile tests that the gRPC client compiles and types are correct
func TestGRPCClientCompile(t *testing.T) {
	// Verify the constructor returns the public interface
	c, err := asr.NewGRPCClient(&asr.Config{
		GRPCAddress:   "localhost:50051",
		Model:         "small",
		Language:      "en",
		ChunkFrames:   150,
		OverlapFrames: 50,
	})
	if err != nil {
		t.Fatalf("NewGRPCClient: %v", err)
	}
	var _ asr.GRPCClient = c
	_ = c.CloseClient()
}

// TestStreamMethodsExist tests that Stream methods exist with correct signatures
func TestStreamMethodsExist(t *testing.T) {
	// Verify method signatures exist via the public interfaces
	var _ asr.Stream = (asr.Stream)(nil)
	var _ asr.GRPCClient = (asr.GRPCClient)(nil)
}

// TestGRPCClientMethodsExist tests that GRPCClient methods exist
func TestGRPCClientMethodsExist(t *testing.T) {
	c, err := asr.NewGRPCClient(&asr.Config{GRPCAddress: "localhost:50051", Model: "small", Language: "en"})
	if err != nil {
		t.Fatalf("NewGRPCClient: %v", err)
	}
	defer c.CloseClient()
	_ = c.NewStream
	_ = c.CloseClient
}
