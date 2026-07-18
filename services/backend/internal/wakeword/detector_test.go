package wakeword

import (
	"encoding/binary"
	"testing"
)

// captureScorer records every window it is asked to score.
type captureScorer struct {
	windows [][]float32
	score   float64
	err     error
}

func (c *captureScorer) Score(window []float32) (float64, error) {
	if c.err != nil {
		return 0, c.err
	}
	cp := make([]float32, len(window))
	copy(cp, window)
	c.windows = append(c.windows, cp)
	return c.score, nil
}
func (c *captureScorer) Close() {}

func frameWithValue(v int16) []byte {
	pcm := make([]byte, 640)
	for i := 0; i < 640; i += 2 {
		binary.LittleEndian.PutUint16(pcm[i:], uint16(v))
	}
	return pcm
}

func testCfg() *Config {
	cfg := &Config{ChunkSamples: 1280, WindowSamples: 2560, Threshold: 0.5}
	cfg.Postproc.PatienceFrames = 2
	cfg.Postproc.RefractoryFrames = 100
	cfg.Gating.PrerollFrames = 50
	cfg.Gating.PartialChunk = "drop"
	return cfg
}

func TestDetectorChunksAndRolls(t *testing.T) {
	sc := &captureScorer{score: 0.1}
	d := newDetector(sc, testCfg())

	// frames of values 1..8: chunk A = frames 1-4, chunk B = frames 5-8
	for v := int16(1); v <= 8; v++ {
		_, scored, err := d.pushFrame(frameWithValue(v))
		if err != nil {
			t.Fatal(err)
		}
		wantScored := v%4 == 0
		if scored != wantScored {
			t.Fatalf("frame %d: scored=%v, want %v", v, scored, wantScored)
		}
	}
	if len(sc.windows) != 2 {
		t.Fatalf("got %d windows, want 2", len(sc.windows))
	}
	// after chunk B the window holds [chunk A | chunk B]
	w := sc.windows[1]
	if w[0] != 1 || w[319] != 1 || w[320] != 2 || w[1279] != 4 {
		t.Fatalf("window head wrong: %v %v %v %v", w[0], w[319], w[320], w[1279])
	}
	if w[1280] != 5 || w[2559] != 8 {
		t.Fatalf("window tail wrong: %v %v", w[1280], w[2559])
	}
}

func TestDetectorResetDropsPartialAndZeroesWindow(t *testing.T) {
	sc := &captureScorer{score: 0.1}
	d := newDetector(sc, testCfg())

	for v := int16(1); v <= 6; v++ { // one full chunk + 2 pending frames
		if _, _, err := d.pushFrame(frameWithValue(v)); err != nil {
			t.Fatal(err)
		}
	}
	d.reset()
	// next 4 frames form a fresh chunk; window before it must be zeros
	for v := int16(10); v <= 13; v++ {
		if _, _, err := d.pushFrame(frameWithValue(v)); err != nil {
			t.Fatal(err)
		}
	}
	if len(sc.windows) != 2 {
		t.Fatalf("got %d windows, want 2", len(sc.windows))
	}
	w := sc.windows[1]
	if w[0] != 0 || w[1279] != 0 {
		t.Fatalf("window not zeroed after reset: %v %v", w[0], w[1279])
	}
	if w[1280] != 10 || w[2559] != 13 {
		t.Fatalf("fresh chunk wrong: %v %v", w[1280], w[2559])
	}
}
