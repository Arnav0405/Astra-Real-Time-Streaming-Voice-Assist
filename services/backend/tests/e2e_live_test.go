// Live end-to-end check: build and run the real `astra` server binary, connect
// over a real socket the way clients/mic/mic_client.py does, stream a recorded
// "Astraa" clip, and require the wake word to fire where the golden fixture
// says it does.
//
// This is the automated twin of the two-terminal manual test in
// clients/mic/README.md — same binary, same flags, same wire protocol — so a
// pass means the shipped server (not just the library) detects the wake word.
// e2e_ww_test.go stays the in-process parity gate against the Python reference.
package tests

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"google.golang.org/protobuf/proto"

	"github.com/arnav/astra/services/backend/internal/pb"
	"github.com/arnav/astra/services/backend/internal/vad"
)

// The -verbose trace lines the server prints; the live test reads detections
// off these because the wire protocol has no server->client wake message yet.
var (
	wakeLine   = regexp.MustCompile(`WAKE .* detected \(frame (\d+)\)`)
	listenLine = "astra listening on"
)

func TestLiveWakeWord(t *testing.T) {
	for _, v := range wwVariants {
		t.Run(v.name, func(t *testing.T) {
			runLiveWakeWord(t, v.name, v.model, v.fixture)
		})
	}
}

func runLiveWakeWord(t *testing.T, name, modelPath, fixture string) {
	pcm, wantWakes := loadWakeFixture(t, name, modelPath, fixture)
	if err := vad.Init(""); err != nil {
		t.Skipf("onnxruntime unavailable: %v", err)
	}

	logs, addr := startServer(t, modelPath)
	streamPCM(t, "ws://"+addr, pcm)

	// The sink scores asynchronously; wait for the expected detections to land
	// (then a beat longer, so an extra false fire still shows up as a failure).
	got := waitForWakes(t, logs, len(wantWakes), 30*time.Second)
	time.Sleep(500 * time.Millisecond)
	got = parseWakes(logs.String())

	if len(got) != len(wantWakes) {
		t.Fatalf("wake detections: got frames %v, want %v\n--- server log ---\n%s",
			got, wantWakes, logs.String())
	}
	for i, want := range wantWakes {
		if got[i] != want {
			t.Errorf("wake %d: fired at frame %d, want %d", i, got[i], want)
		}
	}
}

// loadWakeFixture returns the fixture's PCM and expected wake frames, skipping
// the test when the model or fixture for this variant is not committed yet.
func loadWakeFixture(t *testing.T, name, modelPath, fixture string) ([]byte, []int) {
	t.Helper()
	data, err := os.ReadFile(fixture)
	if os.IsNotExist(err) {
		t.Skipf("%s not generated yet (needs trained %s + recordings)", fixture, name)
	}
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("%s not committed yet", modelPath)
	}
	var golden struct {
		PCM        string `json:"pcm_s16le_base64"`
		WakeFrames []int  `json:"wake_frames"`
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	pcm, err := base64.StdEncoding.DecodeString(golden.PCM)
	if err != nil {
		t.Fatal(err)
	}
	if len(golden.WakeFrames) == 0 {
		t.Fatalf("%s has no wake_frames — nothing to detect", fixture)
	}
	return pcm, golden.WakeFrames
}

// startServer builds ./cmd/astra and runs it against modelPath on a free
// loopback port, returning its (concurrently written) log buffer and address.
// The sidecar config is left to the binary's own model-derived default, so the
// test also covers that wiring.
func startServer(t *testing.T, modelPath string) (*syncBuffer, string) {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "astra")
	build := exec.Command("go", "build", "-o", bin, "./cmd/astra")
	build.Dir = ".." // services/backend
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build astra: %v\n%s", err, out)
	}

	addr := fmt.Sprintf("127.0.0.1:%d", freePort(t))
	// Model path is relative to the tests dir; the server runs from
	// services/backend, where its own asset defaults resolve.
	abs, err := filepath.Abs(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	// -no-asr: transcription needs NAGA_API_KEY and is not what this asserts.
	cmd := exec.Command(bin, "-addr", addr, "-verbose", "-no-asr", "-ww-model", abs)
	cmd.Dir = ".."
	logs := &syncBuffer{}
	cmd.Stdout, cmd.Stderr = logs, logs
	if err := cmd.Start(); err != nil {
		t.Fatalf("start astra: %v", err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	})

	deadline := time.Now().Add(30 * time.Second)
	for !strings.Contains(logs.String(), listenLine) {
		if time.Now().After(deadline) {
			t.Fatalf("server never came up\n--- server log ---\n%s", logs.String())
		}
		time.Sleep(50 * time.Millisecond)
	}
	return logs, addr
}

// streamPCM plays the clip through the real wire protocol: StreamStart, 20 ms
// AudioFrames, StreamStop — the same sequence mic_client.py sends.
func streamPCM(t *testing.T, url string, pcm []byte) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, url, nil)
	if err != nil {
		t.Fatalf("dial %s: %v", url, err)
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

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

	for seq := uint64(0); int(seq+1)*frameBytes <= len(pcm); seq++ {
		off := int(seq) * frameBytes
		send(&pb.ClientMessage{Msg: &pb.ClientMessage_AudioFrame{AudioFrame: &pb.AudioFrame{
			Seq: seq, Pcm: pcm[off : off+frameBytes],
		}}})
	}
	send(&pb.ClientMessage{Msg: &pb.ClientMessage_StreamStop{StreamStop: &pb.StreamStop{}}})
}

func waitForWakes(t *testing.T, logs *syncBuffer, want int, timeout time.Duration) []int {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		got := parseWakes(logs.String())
		if len(got) >= want || time.Now().After(deadline) {
			return got
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func parseWakes(log string) []int {
	var frames []int
	for _, m := range wakeLine.FindAllStringSubmatch(log, -1) {
		n, err := strconv.Atoi(m[1])
		if err != nil {
			continue
		}
		frames = append(frames, n)
	}
	return frames
}

// freePort asks the kernel for a port and immediately releases it. Racy in
// principle; in practice the server binds it milliseconds later.
func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// syncBuffer is an io.Writer the child process writes from its own goroutines
// while the test polls String().
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
