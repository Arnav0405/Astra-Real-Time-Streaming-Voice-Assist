package vad

import (
	"encoding/json"
	"os"
	"testing"
)

func loadConfigT(t *testing.T) *Config {
	t.Helper()
	cfg, err := LoadConfig("../../../../assets/models/vad/vad_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

// TestPostprocGolden replays the Python-generated probability sequence and
// requires exact event parity with postproc.py (decision #10).
func TestPostprocGolden(t *testing.T) {
	data, err := os.ReadFile("testdata/postproc_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Probs        []float64 `json:"probs"`
		Events       [][2]any  `json:"events"` // ["start"|"end", frame]
		FinishEvents [][2]any  `json:"finish_events"`
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}

	pp := newPostprocessor(loadConfigT(t))
	var got []Event
	for _, p := range golden.Probs {
		if e, ok := pp.push(p); ok {
			got = append(got, e)
		}
	}
	finish := pp.finish()

	checkEvents(t, "push", got, golden.Events)
	checkEvents(t, "finish", finish, golden.FinishEvents)
}

func checkEvents(t *testing.T, name string, got []Event, want [][2]any) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d events %v, want %d %v", name, len(got), got, len(want), want)
	}
	for i, w := range want {
		kind, frame := w[0].(string), int(w[1].(float64))
		if string(got[i].Type) != kind || got[i].Frame != frame {
			t.Errorf("%s[%d]: got %v, want (%s, %d)", name, i, got[i], kind, frame)
		}
	}
}
