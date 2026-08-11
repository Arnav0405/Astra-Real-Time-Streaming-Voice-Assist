package endpoint

import (
	"log"
	"sync/atomic"

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
	// speaking: the assistant is playing audio back. This is the only state a
	// barge-in can happen in, and the only one where a bare VAD speech_start
	// arms an utterance even in ArmOnWake mode — interrupting a reply by
	// saying the wake word again is not how people talk.
	speaking
)

// bargeRingSlack pads the barge-in preroll ring beyond the confirmation
// window, covering the frames VAD reaches back over when it retroactively
// declares speech_start.
const bargeRingSlack = 8

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
	onBargeIn   func()

	state       int
	buf         []byte
	frames      int    // frames buffered since arm
	framesAtEnd int    // frames at EventEnd (speech span); 0 until EventEnd
	graceCount  int    // silence frames counted since entering grace
	startSeq    uint64 // seq of first buffered frame
	lastSeq     uint64 // seq of most recent buffered frame

	// Barge-in bookkeeping, live only in the speaking state. The ring keeps
	// the most recent frames so an interruption arrives with its onset
	// intact: the decision that speech is a barge-in is necessarily made
	// BargeInFrames after that speech began.
	ring        [][]byte
	ringSeq     []uint64
	ringNext    int
	ringLen     int
	bargeSpeech bool // VAD currently reports speech during playback
	bargeCount  int  // consecutive speech frames since it started

	// speakingReq is written by the reply runner from its own goroutine and
	// read by the frame goroutine; see SetSpeaking.
	speakingReq atomic.Bool
}

// NewMachine returns a per-stream endpoint machine. A nil onUtterance logs.
// onBargeIn, if non-nil, fires when the user talks over the assistant; it must
// not block, since it runs on the frame path.
func NewMachine(streamID string, cfg *Config, mode Mode, onUtterance func(Utterance), onBargeIn func()) *Machine {
	if onUtterance == nil {
		onUtterance = func(u Utterance) {
			log.Printf("stream %s: utterance seq %d-%d (%d frames)", streamID, u.StartSeq, u.EndSeq, u.FrameCount)
		}
	}
	ringSize := cfg.BargeInFrames + bargeRingSlack
	return &Machine{
		streamID:    streamID,
		cfg:         cfg,
		mode:        mode,
		onUtterance: onUtterance,
		onBargeIn:   onBargeIn,
		ring:        make([][]byte, ringSize),
		ringSeq:     make([]uint64, ringSize),
	}
}

// SetSpeaking reports whether the assistant is currently playing reply audio.
//
// It is the one method safe to call from another goroutine — the reply runner
// owns playback and lives off the frame path. Rather than mutate state from
// there, it records the request and the frame goroutine applies it on the next
// event, so every field of Machine stays owned by a single goroutine.
func (m *Machine) SetSpeaking(v bool) { m.speakingReq.Store(v) }

// applySpeaking reconciles the requested playback state. Called at the top of
// both OnVad and OnFrame so a speech_start arriving in the same frame as the
// transition is not missed.
func (m *Machine) applySpeaking() {
	want := m.speakingReq.Load()
	switch {
	case want && m.state == idle:
		// Only from idle: a turn cannot begin while an utterance is open.
		m.state = speaking
		m.resetBarge()
	case !want && m.state == speaking:
		// A barge-in has already left speaking, so this only unwinds a reply
		// that finished or failed on its own.
		m.state = idle
		m.resetBarge()
	}
}

// OnWake arms the utterance in wake-word mode.
func (m *Machine) OnWake(_ wakeword.Event) {
	if m.mode == ArmOnWake && m.state == idle {
		m.arm()
	}
}

// OnVad reacts to VAD speech boundaries.
func (m *Machine) OnVad(e vad.Event) {
	m.applySpeaking()
	switch e.Type {
	case vad.EventStart:
		switch {
		case m.state == speaking:
			// Candidate barge-in. It is only confirmed once the speech has
			// persisted for BargeInFrames — a cough or a burst of echo the
			// canceller let through must not cut the assistant off.
			m.bargeSpeech = true
			m.bargeCount = 0
		case m.state == grace:
			m.state = capturing // pause within grace → same utterance continues
		case m.state == idle && m.mode == ArmOnVad:
			m.arm()
		}
	case vad.EventEnd:
		switch m.state {
		case speaking:
			m.bargeSpeech = false
			m.bargeCount = 0
		case capturing:
			m.state = grace
			m.graceCount = 0
			m.framesAtEnd = m.frames
		}
	}
}

// OnFrame buffers one frame's PCM and advances the grace / timeout counters.
func (m *Machine) OnFrame(seq uint64, pcm []byte) {
	m.applySpeaking()
	if m.state == speaking {
		m.pushRing(seq, pcm)
		if m.bargeSpeech {
			m.bargeCount++
			if m.bargeCount >= m.cfg.BargeInFrames {
				m.bargeIn()
			}
		}
		return
	}
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

// bargeIn confirms the user is talking over the assistant: it opens an
// utterance seeded with the buffered onset, then notifies the caller so the
// reply can be cancelled.
func (m *Machine) bargeIn() {
	m.arm()
	m.seedFromRing()
	m.resetBarge()
	if m.onBargeIn != nil {
		m.onBargeIn()
	}
}

// pushRing records one frame in the preroll ring.
func (m *Machine) pushRing(seq uint64, pcm []byte) {
	if len(m.ring) == 0 {
		return
	}
	i := m.ringNext
	if len(m.ring[i]) != len(pcm) {
		m.ring[i] = make([]byte, len(pcm))
	}
	copy(m.ring[i], pcm)
	m.ringSeq[i] = seq
	m.ringNext = (m.ringNext + 1) % len(m.ring)
	if m.ringLen < len(m.ring) {
		m.ringLen++
	}
}

// seedFromRing prepends the buffered preroll to a freshly armed utterance, so
// the words that triggered the barge-in are the ones transcribed. Must be
// called immediately after arm(), while the buffer is still empty.
func (m *Machine) seedFromRing() {
	if m.ringLen == 0 {
		return
	}
	start := (m.ringNext - m.ringLen + len(m.ring)) % len(m.ring)
	for i := 0; i < m.ringLen; i++ {
		idx := (start + i) % len(m.ring)
		if i == 0 {
			m.startSeq = m.ringSeq[idx]
		}
		m.buf = append(m.buf, m.ring[idx]...)
		m.lastSeq = m.ringSeq[idx]
	}
	m.frames = m.ringLen
}

func (m *Machine) resetBarge() {
	m.bargeSpeech = false
	m.bargeCount = 0
	m.ringNext = 0
	m.ringLen = 0
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
