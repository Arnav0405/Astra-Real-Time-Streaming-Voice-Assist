package endpoint

import (
	"log"

	"github.com/arnav/astra/services/backend/internal/vad"
	"github.com/arnav/astra/services/backend/internal/wakeword"
)

type Mode int

const (
	// ArmOnVad: no wake word — arm on VAD EventStart (speech onset).
	ArmOnVad Mode = iota
	// ArmOnWake: arm on the wake Event; a bare VAD EventStart never arms from
	// idle (it only reopens during grace).
	ArmOnWake
)

const (
	idle = iota
	capturing
	grace
)

// Utterance is one closed span, handed to the downstream consumer. PCM is a
// private copy (arm..close, includes the grace-silence tail — harmless for
// ASR). FrameCount is the speech span (arm..EventEnd), the value the min-length
// guard is measured against.
type Utterance struct {
	StreamID   string
	StartSeq   uint64
	EndSeq     uint64
	FrameCount int
	PCM        []byte
}

// Machine is the per-stream endpoint state machine. It is fed by the sink via
// OnVad/OnWake (VAD/wake events) and OnFrame (every frame's PCM). Not safe for
// concurrent use — the sink drives it from a single goroutine.
type Machine struct {
	streamID    string
	cfg         *Config
	mode        Mode
	onUtterance func(Utterance)

	state       int
	buf         []byte
	frames      int    // frames buffered since arm
	framesAtEnd int    // frames at EventEnd (speech span); 0 until EventEnd
	graceCount  int    // silence frames counted since entering grace
	startSeq    uint64 // seq of first buffered frame
	lastSeq     uint64 // seq of most recent buffered frame
}

// NewMachine returns a per-stream endpoint machine. A nil onUtterance logs.
func NewMachine(streamID string, cfg *Config, mode Mode, onUtterance func(Utterance)) *Machine {
	if onUtterance == nil {
		onUtterance = func(u Utterance) {
			log.Printf("stream %s: utterance seq %d-%d (%d frames)", streamID, u.StartSeq, u.EndSeq, u.FrameCount)
		}
	}
	return &Machine{streamID: streamID, cfg: cfg, mode: mode, onUtterance: onUtterance}
}

// OnWake arms the utterance in wake-word mode.
func (m *Machine) OnWake(_ wakeword.Event) {
	if m.mode == ArmOnWake && m.state == idle {
		m.arm()
	}
}

// OnVad reacts to VAD speech boundaries.
func (m *Machine) OnVad(e vad.Event) {
	switch e.Type {
	case vad.EventStart:
		switch {
		case m.state == grace:
			m.state = capturing // pause within grace → same utterance continues
		case m.state == idle && m.mode == ArmOnVad:
			m.arm()
		}
	case vad.EventEnd:
		if m.state == capturing {
			m.state = grace
			m.graceCount = 0
			m.framesAtEnd = m.frames
		}
	}
}

// OnFrame buffers one frame's PCM and advances the grace / timeout counters.
func (m *Machine) OnFrame(seq uint64, pcm []byte) {
	if m.state == idle {
		return
	}
	if m.frames == 0 {
		m.startSeq = seq
	}
	m.buf = append(m.buf, pcm...)
	m.frames++
	m.lastSeq = seq

	if m.state == grace {
		m.graceCount++
		if m.graceCount >= m.cfg.GraceFrames {
			m.close()
			return
		}
	}
	if m.frames >= m.cfg.MaxUtteranceFrames {
		m.close()
	}
}

func (m *Machine) arm() {
	m.state = capturing
	m.buf = m.buf[:0]
	m.frames = 0
	m.framesAtEnd = 0
	m.graceCount = 0
}

func (m *Machine) close() {
	n := m.framesAtEnd
	if n == 0 { // max-timeout path: no EventEnd was seen
		n = m.frames
	}
	if n >= m.cfg.MinUtteranceFrames {
		pcm := make([]byte, len(m.buf))
		copy(pcm, m.buf) // m.buf is reused; the consumer must own its copy
		m.onUtterance(Utterance{
			StreamID:   m.streamID,
			StartSeq:   m.startSeq,
			EndSeq:     m.lastSeq,
			FrameCount: n,
			PCM:        pcm,
		})
	}
	m.state = idle
	m.buf = m.buf[:0]
	m.frames = 0
	m.framesAtEnd = 0
	m.graceCount = 0
}
