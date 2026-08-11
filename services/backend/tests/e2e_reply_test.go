// End-to-end checks for Phase 7: stream real speech over WebSocket through the
// full server + VAD + endpointing + ASR + reply pipeline, and require that a
// spoken reply comes back — and that talking over it cuts it off.
//
// The VAD and endpoint machine are the real ones; only the three network
// dependencies (ASR, LLM, TTS) are stubbed, so what is under test is the
// wiring and the barge-in decision, not a provider.
package tests

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"google.golang.org/protobuf/proto"

	"github.com/arnav/astra/services/backend/internal/asr"
	"github.com/arnav/astra/services/backend/internal/endpoint"
	"github.com/arnav/astra/services/backend/internal/pb"
	"github.com/arnav/astra/services/backend/internal/server"
	"github.com/arnav/astra/services/backend/internal/turn"
	"github.com/arnav/astra/services/backend/internal/vad"
)

const replyRate = 24000

// --- stubs ----------------------------------------------------------------

type stubASR struct{ text string }

func (s stubASR) Transcribe([]byte) (string, error) { return s.text, nil }

type stubLLM struct{ deltas []string }

func (s stubLLM) Stream(ctx context.Context, _ string, onDelta func(string)) error {
	for _, d := range s.deltas {
		if err := ctx.Err(); err != nil {
			return err
		}
		onDelta(d)
	}
	return nil
}

// stubTTS emits a long reply in small chunks so a barge-in has something to
// interrupt. It stops the moment the turn is cancelled.
type stubTTS struct{ chunks int }

func (s stubTTS) Speak(ctx context.Context, _ string, onPCM func([]byte) error) error {
	// 100 ms of 24 kHz s16le per chunk.
	chunk := make([]byte, replyRate/10*2)
	for i := 0; i < s.chunks; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := onPCM(chunk); err != nil {
			return err
		}
	}
	return nil
}

// --- harness --------------------------------------------------------------

type replyRig struct {
	t    *testing.T
	conn *websocket.Conn
	ctx  context.Context
	pcm  []byte
	seq  uint64
	sink server.Sink
}

func loadSpeechPCM(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("../internal/vad/testdata/e2e_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		PCM string `json:"pcm_s16le_base64"`
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	pcm, err := base64.StdEncoding.DecodeString(golden.PCM)
	if err != nil {
		t.Fatal(err)
	}
	return pcm
}

func newReplyRig(t *testing.T, llm stubLLM, synth stubTTS) *replyRig {
	t.Helper()
	if err := vad.Init(""); err != nil {
		t.Skipf("onnxruntime unavailable: %v", err)
	}

	vadCfg, err := vad.LoadConfig("../../../assets/models/vad/vad_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	engine, err := vad.NewEngine("../../../assets/models/vad/vad_v1.onnx", vadCfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(engine.Close)

	epCfg, err := endpoint.LoadConfig("../../../assets/configs/endpoint.json")
	if err != nil {
		t.Fatal(err)
	}

	rig := &replyRig{t: t, pcm: loadSpeechPCM(t)}

	srv := server.New()
	srv.NewSink = func(streamID string, send server.Sender) server.Sink {
		var runner *turn.Runner
		worker := asr.NewWorker(streamID, stubASR{text: "what is the weather"}, func(tr asr.Transcript) {
			runner.Start(tr)
		})
		m := endpoint.NewMachine(streamID, epCfg, endpoint.ArmOnVad,
			func(u endpoint.Utterance) { worker.Enqueue(u) },
			func() { runner.Barge() })
		runner = turn.NewRunner(streamID, send, llm, synth, replyRate, m.SetSpeaking)

		rig.sink = vad.NewSink(streamID, engine, vadCfg, m.OnVad, m.OnFrame)
		return rig.sink
	}

	ts := httptest.NewServer(srv)
	t.Cleanup(ts.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	conn, _, err := websocket.Dial(ctx, strings.Replace(ts.URL, "http", "ws", 1), nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close(websocket.StatusNormalClosure, "") })

	rig.conn, rig.ctx = conn, ctx
	rig.send(&pb.ClientMessage{Msg: &pb.ClientMessage_StreamStart{StreamStart: &pb.StreamStart{
		SampleRateHz: 16000, Channels: 1, BitsPerSample: 16, FrameDurationMs: 20,
	}}})
	if got := rig.read(); got.GetStreamStarted() == nil {
		t.Fatalf("expected StreamStarted, got %v", got)
	}
	return rig
}

func (r *replyRig) send(msg *pb.ClientMessage) {
	r.t.Helper()
	raw, err := proto.Marshal(msg)
	if err != nil {
		r.t.Fatal(err)
	}
	if err := r.conn.Write(r.ctx, websocket.MessageBinary, raw); err != nil {
		r.t.Fatal(err)
	}
}

func (r *replyRig) read() *pb.ServerMessage {
	r.t.Helper()
	ctx, cancel := context.WithTimeout(r.ctx, 20*time.Second)
	defer cancel()
	_, raw, err := r.conn.Read(ctx)
	if err != nil {
		r.t.Fatalf("read: %v", err)
	}
	var msg pb.ServerMessage
	if err := proto.Unmarshal(raw, &msg); err != nil {
		r.t.Fatal(err)
	}
	return &msg
}

// speech streams the real recorded utterance.
func (r *replyRig) speech() {
	for off := 0; off+frameBytes <= len(r.pcm); off += frameBytes {
		r.send(&pb.ClientMessage{Msg: &pb.ClientMessage_AudioFrame{AudioFrame: &pb.AudioFrame{
			Seq: r.seq, Pcm: r.pcm[off : off+frameBytes],
		}}})
		r.seq++
	}
}

// silence streams n silent frames. n has to be generous: the golden clip ends
// mid-speech at p≈0.999, and the VAD's GRU decays slowly on digital zeros —
// still ≈0.24 after 60 silent frames, against an offset threshold of 0.09. Real
// microphone audio tails off and carries room tone, so it releases far sooner.
func (r *replyRig) silence(n int) {
	quiet := make([]byte, frameBytes)
	for i := 0; i < n; i++ {
		r.send(&pb.ClientMessage{Msg: &pb.ClientMessage_AudioFrame{AudioFrame: &pb.AudioFrame{
			Seq: r.seq, Pcm: quiet,
		}}})
		r.seq++
	}
}

// readUntil reads until pred matches, collecting every message seen.
func (r *replyRig) readUntil(what string, pred func(*pb.ServerMessage) bool) []*pb.ServerMessage {
	r.t.Helper()
	var seen []*pb.ServerMessage
	for i := 0; i < 2000; i++ {
		m := r.read()
		seen = append(seen, m)
		if pred(m) {
			return seen
		}
	}
	r.t.Fatalf("never saw %s after %d messages", what, len(seen))
	return nil
}

// --- tests ----------------------------------------------------------------

// The full loop: speech in, transcript and spoken reply back out.
func TestEndToEndSpokenReply(t *testing.T) {
	rig := newReplyRig(t,
		stubLLM{deltas: []string{"The weather is fine. ", "Anything else? "}},
		stubTTS{chunks: 2})

	rig.speech()
	rig.silence(600) // see silence(): the VAD needs a long tail to release from a clip that ends mid-speech

	msgs := rig.readUntil("ReplyEnd", func(m *pb.ServerMessage) bool { return m.GetReplyEnd() != nil })

	var gotTranscript, gotDelta, gotAudio bool
	var end *pb.ReplyEnd
	for _, m := range msgs {
		switch {
		case m.GetTranscript() != nil:
			gotTranscript = true
			if m.GetTranscript().Text != "what is the weather" {
				t.Errorf("transcript = %q", m.GetTranscript().Text)
			}
		case m.GetReplyDelta() != nil:
			gotDelta = true
		case m.GetReplyAudio() != nil:
			gotAudio = true
			if got := m.GetReplyAudio().SampleRateHz; got != replyRate {
				t.Errorf("reply audio sample_rate_hz = %d, want %d", got, replyRate)
			}
			if len(m.GetReplyAudio().Pcm)%2 != 0 {
				t.Errorf("reply audio chunk of %d bytes splits an s16le sample", len(m.GetReplyAudio().Pcm))
			}
		case m.GetReplyEnd() != nil:
			end = m.GetReplyEnd()
		}
	}

	if !gotTranscript || !gotDelta || !gotAudio {
		t.Fatalf("incomplete reply: transcript=%v delta=%v audio=%v", gotTranscript, gotDelta, gotAudio)
	}
	if end.Reason != pb.ReplyEnd_DONE {
		t.Errorf("reason = %v, want DONE", end.Reason)
	}
}

// Talking over the assistant cuts it off: a Cancel reaches the client (so it
// flushes what it has buffered) and the turn closes as BARGED_IN.
func TestEndToEndBargeIn(t *testing.T) {
	// A long reply — 60 s of audio — so it is certainly still playing when the
	// interruption arrives. Pacing keeps the server from racing ahead of it.
	rig := newReplyRig(t,
		stubLLM{deltas: []string{"Let me tell you a very long story. "}},
		stubTTS{chunks: 600})

	rig.speech()
	rig.silence(600)

	// Wait until the assistant is actually speaking; only then is barge-in
	// possible, and only then has the machine entered the speaking state.
	rig.readUntil("first ReplyAudio", func(m *pb.ServerMessage) bool { return m.GetReplyAudio() != nil })

	// Now talk over it.
	rig.speech()

	msgs := rig.readUntil("Cancel", func(m *pb.ServerMessage) bool { return m.GetCancel() != nil })
	cancelled := msgs[len(msgs)-1].GetCancel()

	end := rig.readUntil("ReplyEnd", func(m *pb.ServerMessage) bool { return m.GetReplyEnd() != nil })
	last := end[len(end)-1].GetReplyEnd()

	if last.Reason != pb.ReplyEnd_BARGED_IN {
		t.Errorf("reason = %v, want BARGED_IN", last.Reason)
	}
	if cancelled.UtteranceId != last.UtteranceId {
		t.Errorf("cancel is for utterance %d but the turn that ended was %d",
			cancelled.UtteranceId, last.UtteranceId)
	}
}
