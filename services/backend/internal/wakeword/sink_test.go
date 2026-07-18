package wakeword

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"testing"

	"github.com/arnav/astra/services/backend/internal/server"
	"github.com/arnav/astra/services/backend/internal/vad"
)

type gatingGolden struct {
	Postproc struct {
		Threshold        float64 `json:"threshold"`
		PatienceFrames   int     `json:"patience_frames"`
		RefractoryFrames int     `json:"refractory_frames"`
	} `json:"postproc"`
	Gating struct {
		PrerollFrames int    `json:"preroll_frames"`
		PartialChunk  string `json:"partial_chunk"`
	} `json:"gating"`
	NFrames   int    `json:"n_frames"`
	PCMBase64 string `json:"pcm_s16le_base64"`
	VadEvents []struct {
		DeliverAt int    `json:"deliver_at"`
		Type      string `json:"type"`
		Frame     int    `json:"frame"`
	} `json:"vad_events"`
	Scores         []float64 `json:"scores"`
	ExpectedChunks []struct {
		Frame    int     `json:"frame"`
		Checksum int64   `json:"checksum"`
		Score    float64 `json:"score"`
	} `json:"expected_chunks"`
	WakeFrames []int `json:"wake_frames"`
}

// scriptedVad delivers golden events keyed by frame arrival index.
type scriptedVad struct {
	events map[int]vad.Event
	frame  int
}

func (v *scriptedVad) Push(pcm []byte) (vad.Event, bool, error) {
	e, ok := v.events[v.frame]
	v.frame++
	return e, ok, nil
}
func (v *scriptedVad) Finish() []vad.Event { return nil }
func (v *scriptedVad) Close()              {}

// scriptedScorer returns golden scores in order and records the checksum of
// the newest chunk (the window's last chunk_samples, int16-valued floats).
type scriptedScorer struct {
	scores    []float64
	chunk     int
	checksums []int64
}

func (s *scriptedScorer) Score(window []float32) (float64, error) {
	var sum int64
	for _, v := range window[len(window)-1280:] {
		sum += int64(v)
	}
	s.checksums = append(s.checksums, ((sum%(1<<31))+(1<<31))%(1<<31))
	if s.chunk >= len(s.scores) {
		return 0, errors.New("scorer ran out of scripted scores")
	}
	score := s.scores[s.chunk]
	s.chunk++
	return score, nil
}
func (s *scriptedScorer) Close() {}

func loadGatingGolden(t *testing.T) *gatingGolden {
	t.Helper()
	data, err := os.ReadFile("testdata/ww_gating_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g gatingGolden
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}
	return &g
}

func gatingSink(g *gatingGolden, vd vadDetector, sc scorer) (*Sink, *[]vad.Event, *[]Event) {
	cfg := &Config{ChunkSamples: 1280, WindowSamples: 2560, Threshold: g.Postproc.Threshold}
	cfg.Postproc.PatienceFrames = g.Postproc.PatienceFrames
	cfg.Postproc.RefractoryFrames = g.Postproc.RefractoryFrames
	cfg.Gating = g.Gating

	vadEvents := &[]vad.Event{}
	wakes := &[]Event{}
	s := &Sink{
		streamID: "test",
		newParts: func() (vadDetector, *detector, error) { return vd, newDetector(sc, cfg), nil },
		cfg:      cfg,
		ringSize: cfg.Gating.PrerollFrames + 4 + 8, // vad_v1 min_speech_frames = 4
		onVad:    func(e vad.Event) { *vadEvents = append(*vadEvents, e) },
		onWake:   func(e Event) { *wakes = append(*wakes, e) },
		done:     make(chan struct{}),
		fatal:    make(chan struct{}),
	}
	return s, vadEvents, wakes
}

func TestSinkGatingGolden(t *testing.T) {
	g := loadGatingGolden(t)
	pcm, err := base64.StdEncoding.DecodeString(g.PCMBase64)
	if err != nil {
		t.Fatal(err)
	}

	events := make(map[int]vad.Event, len(g.VadEvents))
	for _, e := range g.VadEvents {
		events[e.DeliverAt] = vad.Event{Type: vad.EventType(e.Type), Frame: e.Frame}
	}
	sc := &scriptedScorer{scores: g.Scores}
	s, vadGot, wakes := gatingSink(g, &scriptedVad{events: events}, sc)

	frames := make(chan server.Frame, g.NFrames)
	for i := 0; i < g.NFrames; i++ {
		frames <- server.Frame{Seq: uint64(i), PCM: pcm[i*640 : (i+1)*640]}
	}
	close(frames)
	go s.Run(frames)
	s.Wait()

	if len(sc.checksums) != len(g.ExpectedChunks) {
		t.Fatalf("scored %d chunks, want %d", len(sc.checksums), len(g.ExpectedChunks))
	}
	for i, want := range g.ExpectedChunks {
		if sc.checksums[i] != want.Checksum {
			t.Fatalf("chunk %d: checksum %d, want %d", i, sc.checksums[i], want.Checksum)
		}
	}
	if len(*wakes) != len(g.WakeFrames) {
		t.Fatalf("wakes: got %v, want %v", *wakes, g.WakeFrames)
	}
	for i, want := range g.WakeFrames {
		if (*wakes)[i].Frame != want {
			t.Fatalf("wake %d: frame %d, want %d", i, (*wakes)[i].Frame, want)
		}
	}
	if len(*vadGot) != len(g.VadEvents) {
		t.Fatalf("vad events forwarded: got %d, want %d", len(*vadGot), len(g.VadEvents))
	}
}

// failingScorer always errors.
type failingScorer struct{}

func (failingScorer) Score([]float32) (float64, error) { return 0, errors.New("boom") }
func (failingScorer) Close()                           {}

func TestSinkFatalAfterConsecutiveScoreFailures(t *testing.T) {
	g := loadGatingGolden(t)
	events := map[int]vad.Event{0: {Type: vad.EventStart, Frame: 0}}
	s, _, _ := gatingSink(g, &scriptedVad{events: events}, failingScorer{})

	n := 4 * (maxConsecutiveFailures + 5)
	frames := make(chan server.Frame, n)
	for i := 0; i < n; i++ {
		frames <- server.Frame{Seq: uint64(i), PCM: make([]byte, 640)}
	}
	close(frames)
	go s.Run(frames)
	s.Wait()

	select {
	case <-s.Fatal():
	default:
		t.Fatal("expected fatal after consecutive score failures")
	}
	if s.Err() == nil {
		t.Fatal("expected error")
	}
}
