package wakeword

import (
	"os"
	"testing"
)

// TestShippedSidecarRefractoryIsUsable guards the unit trap in the sidecar's
// refractory_frames: it is 20 ms transport frames, but the source config
// (services/ml/configs/ww_v1.yaml) expresses it in *seconds* and the exporter
// multiplies by 50. A plausible frame count read as seconds ships a lockout
// that swallows every wake after the first.
func TestShippedSidecarRefractoryIsUsable(t *testing.T) {
	if _, err := os.Stat(sidecarPath); os.IsNotExist(err) {
		t.Skip("ww_v1.json not committed yet")
	}
	cfg, err := LoadConfig(sidecarPath)
	if err != nil {
		t.Fatal(err)
	}
	const maxFrames = 500 // 10 s: nobody waits longer than this to re-wake
	if cfg.Postproc.RefractoryFrames > maxFrames {
		t.Errorf("refractory_frames = %d (%.1fs); want <= %d (%.1fs) — a longer "+
			"lockout makes the wake word fire once and never again",
			cfg.Postproc.RefractoryFrames, float64(cfg.Postproc.RefractoryFrames)*0.02,
			maxFrames, float64(maxFrames)*0.02)
	}
}
