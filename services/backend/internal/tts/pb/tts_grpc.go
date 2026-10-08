package pb

import (
	grpc "google.golang.org/grpc"
)

// TTSServer is the server API the py Piper servicer implements. Generated as
// a hand-written companion to tts.pb.go (protoc-gen-go only, no -go-grpc),
// same split the py side keeps with tts_pb2_grpc.py.
type TTSServer interface {
	// Synthesize handles exactly one request (the client half-closes after
	// SendMsg) and streams AudioStart then AudioChunk messages back.
	Synthesize(*SynthesizeRequest, TTS_SynthesizeServer) error
}

// TTS_SynthesizeServer is the server-streaming half of Synthesize.
type TTS_SynthesizeServer interface {
	Send(*SynthesizeResponse) error
	grpc.ServerStream
}

// RegisterTTSServer registers a TTSServer on srv under /astra.v1.TTS.
func RegisterTTSServer(srv *grpc.Server, impl TTSServer) {
	srv.RegisterService(&_TTS_serviceDesc, impl)
}

type ttsSynthesizeServer struct{ grpc.ServerStream }

func (s *ttsSynthesizeServer) Send(m *SynthesizeResponse) error {
	return s.ServerStream.SendMsg(m)
}

func _TTS_Synthesize_Handler(srv interface{}, stream grpc.ServerStream) error {
	in := new(SynthesizeRequest)
	if err := stream.RecvMsg(in); err != nil {
		return err
	}
	return srv.(TTSServer).Synthesize(in, &ttsSynthesizeServer{stream})
}

var _TTS_serviceDesc = grpc.ServiceDesc{
	ServiceName: "astra.v1.TTS",
	HandlerType: (*TTSServer)(nil),
	Methods:     []grpc.MethodDesc{},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "Synthesize",
			Handler:       _TTS_Synthesize_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "astra/v1/tts.proto",
}
