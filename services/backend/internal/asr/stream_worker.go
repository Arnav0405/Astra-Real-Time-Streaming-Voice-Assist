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

	// Channels
	frameCh    chan frameMsg
	controlCh  chan controlMsg
	doneCh     chan struct{}

	// Config
	chunkFrames  int
	overlapFrames int
}

type frameMsg struct {
	seq uint64
	pcm []byte
}

type controlMsg struct {
	// Type: 0=arm, 1=disarm, 2=barge-in
	cmd  int
	seq  uint64
	pcm  []byte
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
		frameCh:       make(chan frameMsg, 500), // ~10 seconds buffer
		controlCh:     make(chan controlMsg, 10),
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

// PushFrame pushes a frame to the streaming ASR. Call from the frame path.
func (w *StreamWorker) PushFrame(seq uint64, pcm []byte) {
	if !w.running.Load() {
		return
	}
	select {
	case w.frameCh <- frameMsg{seq: seq, pcm: pcm}:
	default:
		// Drop frame if queue full - logging
		log.Printf("stream %s: stream worker frame queue full, dropping frame %d", w.streamID, seq)
	}
}

// Arm arms the worker for a new utterance. Called when VAD detects speech start.
func (w *StreamWorker) Arm(seq uint64) {
	if !w.running.Load() {
		return
	}
	select {
	case w.controlCh <- controlMsg{cmd: 0, seq: seq}:
	default:
	}
}

// Disarm disarms the worker (end of utterance). Called when VAD detects speech end + grace.
func (w *StreamWorker) Disarm(seq uint64) {
	if !w.running.Load() {
		return
	}
	select {
	case w.controlCh <- controlMsg{cmd: 1, seq: seq}:
	default:
	}
}

// BargeIn handles a barge-in: disarms current utterance and arms new one with buffered frames.
func (w *StreamWorker) BargeIn(seq uint64, pcm []byte) {
	if !w.running.Load() {
		return
	}
	select {
	case w.controlCh <- controlMsg{cmd: 2, seq: seq, pcm: pcm}:
	default:
	}
}

// Close stops the worker and drains in-flight work. Blocks until done.
func (w *StreamWorker) Close() {
	if !w.running.Swap(false) {
		return // already closed
	}
	close(w.frameCh)
	close(w.controlCh)
	<-w.doneCh
}

// run is the main worker loop.
func (w *StreamWorker) run() {
	defer close(w.doneCh)

	var frameBuffer []byte
	var frameSeqs []uint64
	var currentStartSeq uint64
	var frameCount int

	for {
		select {
		case f, ok := <-w.frameCh:
			if !ok {
				return // closed
			}
			w.handleFrame(f, &frameBuffer, &frameSeqs, &currentStartSeq, &frameCount)

		case c, ok := <-w.controlCh:
			if !ok {
				return
			}
			w.handleControl(c, &frameBuffer, &frameSeqs, &currentStartSeq, &frameCount)
		}
	}
}

func (w *StreamWorker) handleFrame(f frameMsg, frameBuffer *[]byte, frameSeqs *[]uint64, currentStartSeq *uint64, frameCount *int) {
	if !w.armed.Load() {
		return
	}

	// Buffer frame locally for potential re-send on barge-in
	*frameBuffer = append(*frameBuffer, f.pcm...)
	*frameSeqs = append(*frameSeqs, f.seq)
	if *frameCount == 0 {
		*currentStartSeq = f.seq
	}
	*frameCount++

	// Send to gRPC stream
	w.mu.Lock()
	stream := w.stream
	w.mu.Unlock()

	if stream != nil {
		if *frameCount <= 5 || *frameCount%100 == 0 {
			log.Printf("stream %s: pushing frame %d (total pushed: %d)", w.streamID, f.seq, *frameCount)
		}
		if err := stream.PushPCM(f.pcm); err != nil {
			log.Printf("stream %s: push frame %d error: %v", w.streamID, f.seq, err)
			// Stream broken - will be handled by receiver
		}
	} else {
		if *frameCount <= 5 {
			log.Printf("stream %s: NO STREAM to push frame %d", w.streamID, f.seq)
		}
	}
}

func (w *StreamWorker) handleControl(c controlMsg, frameBuffer *[]byte, frameSeqs *[]uint64, currentStartSeq *uint64, frameCount *int) {
	w.mu.Lock()
	defer w.mu.Unlock()

	switch c.cmd {	case 0: // Arm
		if w.armed.Load() {
			// Already armed - might be a pause within grace, continue
			return
		}
		log.Printf("stream %s: arming utterance %d", w.streamID, w.utteranceID+1)
		w.armed.Store(true)
		w.utteranceID++

		// Create new gRPC stream for this utterance
		ctx := context.Background()
		stream, err := w.grpcClient.NewStream(ctx, w.utteranceID)
		if err != nil {
			log.Printf("stream %s: failed to create ASR stream: %v", w.streamID, err)
			w.armed.Store(false)
			return
		}
		log.Printf("stream %s: gRPC stream created, utteranceID=%d", w.streamID, w.utteranceID)
		w.stream = stream
		w.armed.Store(true)
		resetUtterance(frameBuffer, frameSeqs, frameCount)

		// Start receiver goroutine
		go w.recvLoop(stream, *currentStartSeq)

	case 1: // Disarm
		if !w.armed.Load() {
			return
		}
		w.armed.Store(false)
		if w.stream != nil {
			w.stream.Close()
			w.stream = nil
		}

		// Emit final transcript (will come from recvLoop)
		// The final transcript is emitted when we receive is_final=true

	case 2: // Barge-in
		// Close current stream
		if w.stream != nil {
			w.stream.Close()
			w.stream = nil
		}
		w.armed.Store(false)

		// Start new utterance with buffered frames
		w.utteranceID++
		ctx := context.Background()
		stream, err := w.grpcClient.NewStream(ctx, w.utteranceID)
		if err != nil {
			log.Printf("stream %s: failed to create ASR stream after barge-in: %v", w.streamID, err)
			return
		}
		w.stream = stream
		w.armed.Store(true)
		resetUtterance(frameBuffer, frameSeqs, frameCount)

		// The preroll is one concatenated PCM blob: push it as a single chunk.
		if len(c.pcm) > 0 {
			if err := stream.PushPCM(c.pcm); err != nil {
				log.Printf("stream %s: push barge-in preroll error: %v", w.streamID, err)
			}
		}

		go w.recvLoop(stream, c.seq)
	}
}

// resetUtterance clears the per-utterance frame bookkeeping so counters and
// buffers restart with each utterance instead of growing across them.
func resetUtterance(frameBuffer *[]byte, frameSeqs *[]uint64, frameCount *int) {
	*frameBuffer = (*frameBuffer)[:0]
	*frameSeqs = (*frameSeqs)[:0]
	*frameCount = 0
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