package wakeword

import (
	"encoding/json"
	"os"
	"testing"
)

type postprocGolden struct {
	Postproc struct {
		Threshold        float64 `json:"threshold"`
		PatienceFrames   int     `json:"patience_frames"`
		RefractoryFrames int     `json:"refractory_frames"`
	} `json:"postproc"`
	Steps []struct {
		Score           float64 `json:"score"`
		Frame           int     `json:"frame"`
		GateResetBefore bool    `json:"gate_reset_before"`
	} `json:"steps"`
	Triggers []int `json:"triggers"`
}

func TestPostprocGolden(t *testing.T) {
	data, err := os.ReadFile("testdata/ww_postproc_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g postprocGolden
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}

	cfg := &Config{Threshold: g.Postproc.Threshold}
	cfg.Postproc.PatienceFrames = g.Postproc.PatienceFrames
	cfg.Postproc.RefractoryFrames = g.Postproc.RefractoryFrames

	pp := newPostprocessor(cfg)
	var got []int
	for _, s := range g.Steps {
		if s.GateResetBefore {
			pp.gateReset()
		}
		if frame, ok := pp.push(s.Score, s.Frame); ok {
			got = append(got, frame)
		}
	}

	if len(got) != len(g.Triggers) {
		t.Fatalf("triggers: got %v, want %v", got, g.Triggers)
	}
	for i := range got {
		if got[i] != g.Triggers[i] {
			t.Fatalf("trigger %d: got frame %d, want %d", i, got[i], g.Triggers[i])
		}
	}
}
