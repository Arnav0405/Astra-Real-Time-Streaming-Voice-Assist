package tests

import (
	"testing"

	"github.com/arnav/astra/services/backend/internal/asr"
)

// TestGRPCClientCompile tests that the gRPC client compiles and types are correct
func TestGRPCClientCompile(t *testing.T) {
	// Just verify the interface and types compile correctly
	var _ asr.GRPCClient = (*asr.grpcClient)(nil)
	var _ asr.Stream = (*asr.grpcStream)(nil)

	// Verify config struct has expected fields
	cfg := &asr.Config{
		GRPCAddress:    "localhost:50051",
		Model:          "small",
		Language:       "en",
		ChunkFrames:    150,
		OverlapFrames:  50,
	}
	_ = cfg
}

// TestStreamMethodsExist tests that Stream methods exist with correct signatures
func TestStreamMethodsExist(t *testing.T) {
	// Verify method signatures exist by taking their addresses
	_ = (*asr.grpcStream).PushPCM
	_ = (*asr.grpcStream).Recv
	_ = (*asr.grpcStream).Close
}

// TestGRPCClientMethodsExist tests that GRPCClient methods exist
func TestGRPCClientMethodsExist(t *testing.T) {
	_ = (*asr.grpcClient).NewStream
	_ = (*asr.grpcClient).CloseClient
}
