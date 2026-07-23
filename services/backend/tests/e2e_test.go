// End-to-end check for Phase 3: stream a real speech chunk over WebSocket
// through the full server + VAD pipeline and require the exact segment events
// the Python reference produced (testdata/e2e_golden.json).
package tests

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"google.golang.org/protobuf/proto"

	"github.com/arnav/astra/services/backend/internal/pb"
	"github.com/arnav/astra/services/backend/internal/server"
	"github.com/arnav/astra/services/backend/internal/vad"
)

const frameBytes = 640

func TestEndToEndVad(t *testing.T) {
	if err := vad.Init(""); err != nil {
		t.Skipf("onnxruntime unavailable: %v", err)
	}

	data, err := os.ReadFile("../internal/vad/testdata/e2e_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		PCM    string   `json:"pcm_s16le_base64"`
		Events [][2]any `json:"events"`
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	pcm, err := base64.StdEncoding.DecodeString(golden.PCM)
	if err != nil {
		t.Fatal(err)
	}

	cfg, err := vad.LoadConfig("../../../assets/models/vad/vad_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	engine, err := vad.NewEngine("../../../assets/models/vad/vad_v1.onnx", cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()

	var mu sync.Mutex
	var events []vad.Event
	var sink server.Sink
	srv := server.New()
	srv.NewSink = func(streamID string) server.Sink {
		sink = vad.NewSink(streamID, engine, cfg, func(e vad.Event) {
			mu.Lock()
			events = append(events, e)
			mu.Unlock()
		}, nil)
		return sink
	}

	ts := httptest.NewServer(srv)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, strings.Replace(ts.URL, "http", "ws", 1), nil)
	if err != nil {
		t.Fatal(err)
	}

	send := func(msg *pb.ClientMessage) {
		t.Helper()
		raw, err := proto.Marshal(msg)
		if err != nil {
			t.Fatal(err)
		}
		if err := conn.Write(ctx, websocket.MessageBinary, raw); err != nil {
			t.Fatal(err)
		}
	}

	send(&pb.ClientMessage{Msg: &pb.ClientMessage_StreamStart{StreamStart: &pb.StreamStart{
		SampleRateHz: 16000, Channels: 1, BitsPerSample: 16, FrameDurationMs: 20,
	}}})
	_, raw, err := conn.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var reply pb.ServerMessage
	if err := proto.Unmarshal(raw, &reply); err != nil {
		t.Fatal(err)
	}
	if reply.GetStreamStarted() == nil {
		t.Fatalf("expected StreamStarted, got %v", &reply)
	}

	for seq := uint64(0); int(seq)*frameBytes < len(pcm); seq++ {
		off := int(seq) * frameBytes
		send(&pb.ClientMessage{Msg: &pb.ClientMessage_AudioFrame{AudioFrame: &pb.AudioFrame{
			Seq: seq, Pcm: pcm[off : off+frameBytes],
		}}})
	}
	send(&pb.ClientMessage{Msg: &pb.ClientMessage_StreamStop{StreamStop: &pb.StreamStop{}}})
	conn.Close(websocket.StatusNormalClosure, "")

	sink.Wait()
	if err := sink.Err(); err != nil {
		t.Fatalf("sink failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(events) != len(golden.Events) {
		t.Fatalf("got events %v, want %v", events, golden.Events)
	}
	for i, w := range golden.Events {
		kind, frame := w[0].(string), int(w[1].(float64))
		if string(events[i].Type) != kind || events[i].Frame != frame {
			t.Fatalf("event %d: got %v, want (%s, %d)", i, events[i], kind, frame)
		}
	}
}
