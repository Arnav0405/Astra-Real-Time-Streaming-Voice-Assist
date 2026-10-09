package asr

import (
	"context"
	"log"
	"strings"
	"sync"
	"sync/atomic"
)

// StreamWorker manages the streaming ASR pipeline for one stream.
// It receives frames incrementally, sends them to the gRPC ASR service,
// and emits partial and final transcripts.
type StreamWorker struct {
	streamID  string
	grpcAddr  string
	model     string
	language  string

	mu          sync.Mutex
	grpcClient  GRPCClient
	stream      Stream
	utteranceID uint64

	// Callback for transcripts (partial and final)
	onTranscript func(StreamTranscript)

	// State
	running    atomic.Bool
	armed      atomic.Bool
	graceFrames int

	// queue is the worker's only inbound channel; run() drains it in order.
	queue chan workerMsg
	// sendMu guards queue against Close: sending on a closed channel panics,
	// and the frame path pushes while the session tears the worker down.
	sendMu sync.Mutex
	closed bool
	doneCh chan struct{}

	// Config
	chunkFrames  int
	overlapFrames int
}

// queueDepth bounds the worker's backlog: 500 frames is 10 s of audio, and a
// queue that deep already means ASR is not keeping up.
const queueDepth = 500

// Message kinds on the worker's single queue.
const (
	msgFrame = iota
	msgArm
	msgDisarm
	msgBarge
)

// workerMsg is one entry on the worker queue: a frame, or a control request.
// Frames and control share ONE channel so they are applied in enqueue order.
// Arm and the frame right behind it are both ready at that instant, and a
// select over two channels picks one at random — which silently discarded the
// opening frame of about half of all utterances.
type workerMsg struct {
	kind int
	seq  uint64
	pcm  []byte
}

// Transcript is one transcribed utterance, handed to the downstream consumer
// (the LLM reply runner). Built from a StreamTranscript in the streaming path.
type Transcript struct {
	StreamID   string
	Text       string
	StartSeq   uint64
	EndSeq     uint64
	FrameCount int
}

// StreamTranscript is a partial or final transcript from streaming ASR.
type StreamTranscript struct {
	StreamID   string
	Text       string
	IsFinal    bool
	StartSeq   uint64
	EndSeq     uint64
	FrameCount int
}

// StreamWorkerConfig holds configuration for the streaming worker.
type StreamWorkerConfig struct {
	GRPCAddress   string
	Model         string
	Language      string
	ChunkFrames   int
	OverlapFrames int
	GraceFrames   int
}

// NewStreamWorker creates a new streaming ASR worker.
func NewStreamWorker(streamID string, cfg *StreamWorkerConfig, onTranscript func(StreamTranscript)) *StreamWorker {
	if onTranscript == nil {
		onTranscript = func(t StreamTranscript) {
			log.Printf("stream %s: transcript final=%v seq %d-%d: %q", t.StreamID, t.IsFinal, t.StartSeq, t.EndSeq, t.Text)
		}
	}

	w := &StreamWorker{
		streamID:      streamID,
		grpcAddr:      cfg.GRPCAddress,
		model:         cfg.Model,
		language:      cfg.Language,
		onTranscript:  onTranscript,
		chunkFrames:   cfg.ChunkFrames,
		overlapFrames: cfg.OverlapFrames,
		graceFrames:   cfg.GraceFrames,
		queue:         make(chan workerMsg, queueDepth),
		doneCh:        make(chan struct{}),
	}

	return w
}

// Start starts the streaming worker. Must be called before PushFrame.
func (w *StreamWorker) Start() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.running.Load() {
		return nil
	}

	// Create gRPC client
	cfg := &Config{
		GRPCAddress: w.grpcAddr,
		Model:       w.model,
		Language:    w.language,
		ChunkFrames: w.chunkFrames,
		OverlapFrames: w.overlapFrames,
	}
	cl, err := NewGRPCClient(cfg)
	if err != nil {
		return err
	}
	w.grpcClient = cl

	w.running.Store(true)
	go w.run()

	return nil
}

// enqueue hands one message to the worker goroutine without blocking. Returns
// false when the worker is shutting down or the queue is full.
func (w *StreamWorker) enqueue(m workerMsg) bool {
	w.sendMu.Lock()
	defer w.sendMu.Unlock()
	if w.closed {
		return false
	}
	select {
	case w.queue <- m:
		return true
	default:
		return false
	}
}

// PushFrame pushes a frame to the streaming ASR. Call from the frame path; it
// never blocks.
func (w *StreamWorker) PushFrame(seq uint64, pcm []byte) {
	if !w.running.Load() {
		return
	}
	w.enqueue(workerMsg{kind: msgFrame, seq: seq, pcm: pcm})
}

// Arm arms the worker for a new utterance (VAD speech start, or a wake word).
func (w *StreamWorker) Arm(seq uint64) {
	if !w.running.Load() {
		return
	}
	w.enqueue(workerMsg{kind: msgArm, seq: seq})
}

// Disarm closes the current utterance's stream, which makes the server finalize
// and emit the final transcript.
func (w *StreamWorker) Disarm(seq uint64) {
	if !w.running.Load() {
		return
	}
	w.enqueue(workerMsg{kind: msgDisarm, seq: seq})
}

// BargeIn closes the current stream and arms a new one seeded with the buffered
// onset, so the interrupting words are the ones transcribed.
func (w *StreamWorker) BargeIn(seq uint64, pcm []byte) {
	if !w.running.Load() {
		return
	}
	w.enqueue(workerMsg{kind: msgBarge, seq: seq, pcm: pcm})
}

// Close stops the worker and drains in-flight work. Blocks until run() has
// drained the queue. Idempotent, and safe when Start() never ran (nothing was
// launched, so doneCh is never closed).
func (w *StreamWorker) Close() {
	if !w.running.Swap(false) {
		return
	}
	w.sendMu.Lock()
	w.closed = true
	close(w.queue)
	w.sendMu.Unlock()
	<-w.doneCh
}

// run is the worker's single goroutine. It drains one queue in order, so a
// control message is always applied before the frames queued behind it.
func (w *StreamWorker) run() {
	defer close(w.doneCh)
	for m := range w.queue {
		switch m.kind {
		case msgFrame:
			w.handleFrame(m)
		default:
			w.handleControl(m)
		}
	}
}

func (w *StreamWorker) handleFrame(m workerMsg) {
	if !w.armed.Load() {
		return
	}
	w.mu.Lock()
	stream := w.stream
	w.mu.Unlock()
	if stream == nil {
		return // arm is ordered ahead of this frame; the stream arrives first
	}
	if err := stream.PushPCM(m.pcm); err != nil {
		log.Printf("stream %s: push frame %d error: %v", w.streamID, m.seq, err)
	}
}

func (w *StreamWorker) handleControl(m workerMsg) {
	w.mu.Lock()
	defer w.mu.Unlock()

	switch m.kind {
	case msgArm:
		if w.armed.Load() {
			return
		}
		w.utteranceID++
		stream, err := w.grpcClient.NewStream(context.Background(), w.utteranceID)
		if err != nil {
			log.Printf("stream %s: failed to create ASR stream: %v", w.streamID, err)
			w.armed.Store(false)
			return
		}
		log.Printf("stream %s: gRPC stream created, utteranceID=%d", w.streamID, w.utteranceID)
		w.stream = stream
		w.armed.Store(true)
		go w.recvLoop(stream, m.seq)
	case msgDisarm:
		if !w.armed.Load() {
			return
		}
		w.armed.Store(false)
		if w.stream != nil {
			w.stream.Close()
			w.stream = nil
		}
	case msgBarge:
		if w.stream != nil {
			w.stream.Close()
			w.stream = nil
		}
		w.armed.Store(false)
		w.utteranceID++
		stream, err := w.grpcClient.NewStream(context.Background(), w.utteranceID)
		if err != nil {
			log.Printf("stream %s: failed to create ASR stream after barge-in: %v", w.streamID, err)
			return
		}
		w.stream = stream
		w.armed.Store(true)
		if len(m.pcm) > 0 {
			if err := stream.PushPCM(m.pcm); err != nil {
				log.Printf("stream %s: push barge-in preroll error: %v", w.streamID, err)
			}
		}
		go w.recvLoop(stream, m.seq)
	}
}

func (w *StreamWorker) recvLoop(stream Stream, startSeq uint64) {
	log.Printf("stream %s: recvLoop started", w.streamID)
	var running string
	for {
		text, isFinal, err := stream.Recv()
		if err != nil {
			log.Printf("stream %s: recv error: %v", w.streamID, err)
			return
		}
		if text != "" {
			running = stitch(running, text)
		}
		if running == "" {
			continue
		}
		log.Printf("stream %s: got transcript: %q (final=%v)", w.streamID, running, isFinal)

		// Get end sequence (approximate)
		endSeq := startSeq + uint64(w.chunkFrames) // rough estimate

		w.onTranscript(StreamTranscript{
			StreamID:   w.streamID,
			Text:       running,
			IsFinal:    isFinal,
			StartSeq:   startSeq,
			EndSeq:     endSeq,
			FrameCount: w.chunkFrames,
		})

		if isFinal {
			log.Printf("stream %s: recvLoop ended (final received)", w.streamID)
			return
		}
	}
}

// stitch appends delta to the running transcript, dropping the delta's
// leading words that repeat the running text's trailing words. Consecutive
// Whisper chunks share overlap_frames of audio, so each chunk re-transcribes
// the tail of the previous one; a word-level suffix/prefix match removes the
// duplication.
func stitch(running, delta string) string {
	delta = strings.TrimSpace(delta)
	if delta == "" {
		return running
	}
	if running == "" {
		return delta
	}
	rWords := strings.Fields(running)
	dWords := strings.Fields(delta)
	max := len(dWords)
	if len(rWords) < max {
		max = len(rWords)
	}
	overlap := 0
	for k := max; k > 0; k-- {
		if sameWords(rWords[len(rWords)-k:], dWords[:k]) {
			overlap = k
			break
		}
	}
	rest := strings.Join(dWords[overlap:], " ")
	if rest == "" {
		return running
	}
	return running + " " + rest
}

func sameWords(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !strings.EqualFold(a[i], b[i]) {
			return false
		}
	}
	return true
}

// UpdateGRPCConfig updates the gRPC client config (for reconnection scenarios).
func (w *StreamWorker) UpdateGRPCConfig(addr, model, language string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.grpcAddr = addr
	w.model = model
	w.language = language
	// Next arm will use new config
}