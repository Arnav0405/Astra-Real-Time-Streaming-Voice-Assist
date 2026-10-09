package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/joho/godotenv"

	"github.com/arnav/astra/services/backend/internal/asr"
	"github.com/arnav/astra/services/backend/internal/endpoint"
	"github.com/arnav/astra/services/backend/internal/llm"
	"github.com/arnav/astra/services/backend/internal/metrics"
	"github.com/arnav/astra/services/backend/internal/pb"
	"github.com/arnav/astra/services/backend/internal/server"
	"github.com/arnav/astra/services/backend/internal/tts"
	"github.com/arnav/astra/services/backend/internal/turn"
	"github.com/arnav/astra/services/backend/internal/vad"
	"github.com/arnav/astra/services/backend/internal/wakeword"
)

type drainSink struct {
	server.Sink
	streamWorker *asr.StreamWorker
	runner       *turn.Runner
}

func (s drainSink) Run(frames <-chan server.Frame) {
	s.Sink.Run(frames)
	go func() {
		if s.runner != nil {
			s.runner.Close()
		}
		if s.streamWorker != nil {
			s.streamWorker.Close()
		}
	}()
}

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	ortLib := flag.String("ort-lib", "", "path to onnxruntime shared library (default: $ASTRA_ORT_LIB, then well-known paths)")
	vadModel := flag.String("vad-model", "../../assets/models/vad/vad_v1.onnx", "path to VAD ONNX model (default assumes running from services/backend)")
	vadConfig := flag.String("vad-config", "../../assets/models/vad/vad_v1.json", "path to VAD sidecar config")
	wwModel := flag.String("ww-model", "../../assets/models/wakeword/ww_v2.onnx", "path to merged wake-word ONNX model; empty disables wake word (VAD-only sink)")
	wwConfig := flag.String("ww-config", "", "path to wake-word sidecar config (default: the model path with .onnx -> .json)")
	endpointConfig := flag.String("endpoint-config", "../../assets/configs/endpoint.json", "path to endpoint (utterance) config")
	endpointWavDir := flag.String("endpoint-wav-dir", "", "if set, dump each closed utterance here as WAV (debug/verification)")
	asrConfig := flag.String("asr-config", "../../assets/configs/asr.json", "path to ASR (transcription) config")
	noASR := flag.Bool("no-asr", false, "disable transcription (VAD/wake/endpointing only)")
	llmConfig := flag.String("llm-config", "../../assets/configs/llm.json", "path to LLM (reply) config")
	ttsConfig := flag.String("tts-config", "../../assets/configs/tts.json", "path to TTS (speech synthesis) config")
	noReply := flag.Bool("no-reply", false, "disable the spoken reply (transcribe only, no LLM/TTS/barge-in)")
	envFile := flag.String("env-file", "../../.env", "path to .env file with OPENCODE_GO_KEY (already-exported env wins)")
	webDir := flag.String("web-dir", "../../clients/web", "directory served at /app/ (the browser demo client); empty disables it")
	verbose := flag.Bool("verbose", false, "log the per-stream pipeline trace (VAD speech/silence, wake, utterance)")
	flag.Parse()

	godotenv.Load(*envFile) // load .env early so env vars like ASTRA_ORT_LIB are available
	if err := vad.Init(*ortLib); err != nil {
		log.Fatal(err)
	}
	cfg, err := vad.LoadConfig(*vadConfig)
	if err != nil {
		log.Fatal(err)
	}
	engine, err := vad.NewEngine(*vadModel, cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	epCfg, err := endpoint.LoadConfig(*endpointConfig)
	if err != nil {
		log.Fatal(err)
	}

	var asrEnabled bool
	var asrGRPCAddr, asrModel, asrLanguage string
	var asrChunkFrames, asrOverlapFrames, asrGraceFrames int
	if *noASR {
		log.Print("asr disabled (-no-asr): utterances will not be transcribed")
	} else {
		acfg, err := asr.LoadConfig(*asrConfig)
		if err != nil {
			log.Fatal(err)
		}
		asrEnabled = true
		asrGRPCAddr = acfg.GRPCAddress
		asrModel = acfg.Model
		asrLanguage = acfg.Language
		asrChunkFrames = acfg.ChunkFrames
		asrOverlapFrames = acfg.OverlapFrames
		asrGraceFrames = acfg.GraceFrames
		log.Printf("asr enabled: grpc %s model %s", asrGRPCAddr, asrModel)
	}

	// The reply needs something to reply to, so it follows ASR: with -no-asr
	// there is no transcript and nothing to say.
	var llmClient *llm.Client
	var ttsClient *tts.Client
	var lcfg llm.Config
	var tcfg tts.Config
	switch {
	case *noReply:
		log.Print("reply disabled (-no-reply): transcripts will not be answered")
	case !asrEnabled:
		log.Print("reply disabled: it needs a transcript, and asr is off")
	default:
		godotenv.Load(*envFile)
		key := os.Getenv("OPENCODE_GO_KEY")
		if key == "" {
			log.Fatalf("OPENCODE_GO_KEY not set (checked environment and %s); pass -no-reply to run without spoken replies", *envFile)
		}
		lcfg, err = llm.LoadConfig(*llmConfig)
		if err != nil {
			log.Fatal(err)
		}
		tcfg, err = tts.LoadConfig(*ttsConfig)
		if err != nil {
			log.Fatal(err)
		}
		llmClient = llm.NewClient(lcfg, key)
		ttsClient, err = tts.NewTTSClient(tcfg)
		if err != nil {
			log.Fatal(err)
		}
		defer ttsClient.CloseClient()
		log.Printf("reply enabled: llm %s, tts grpc %s voice %s @ %d Hz",
			lcfg.Model, tcfg.GRPCAddress, tcfg.Voice, tcfg.SampleRateHz)
	}

	// onUtterance for a stream: optional WAV dumper (-endpoint-wav-dir), then
	// nothing else (the streaming ASR worker handles transcripts directly).
	// Under -verbose, wrap it so the utterance end is always logged.
	makeOnUtterance := func(streamID string, rec *metrics.Recorder) func(endpoint.Utterance) {
		var next func(endpoint.Utterance)
		if *endpointWavDir != "" {
			dump, err := endpoint.NewWavDumper(*endpointWavDir, streamID)
			if err != nil {
				log.Printf("stream %s: wav dumper disabled: %v", streamID, err)
			} else {
				next = func(u endpoint.Utterance) { dump(u) }
			}
		}
		if next == nil && !*verbose {
			return nil
		}
		inner := next
		return func(u endpoint.Utterance) {
			rec.Utterance(u.StartSeq)
			if *verbose {
				log.Printf("[%s] UTTR ⏹ listening end — utterance %d frames (seq %d-%d)",
					streamID, u.FrameCount, u.StartSeq, u.EndSeq)
			}
			if inner != nil {
				inner(u)
			}
		}
	}

	withDrain := func(s server.Sink, sw *asr.StreamWorker, r *turn.Runner) server.Sink {
		if sw == nil && r == nil {
			return s
		}
		return drainSink{Sink: s, streamWorker: sw, runner: r}
	}

	replyEnabled := llmClient != nil && ttsClient != nil

	// buildTurn wires one stream's endpoint machine to its reply runner. The
	// two reference each other by design — the machine cancels the reply on
	// barge-in, the runner tells the machine when playback starts and stops —
	// so the machine is built first and the runner reached through a closure.
	buildTurn := func(streamID string, send server.Sender, mode endpoint.Mode, rec *metrics.Recorder) (*endpoint.Machine, *asr.StreamWorker, *turn.Runner) {
		var runner *turn.Runner
		// Every outbound message passes the recorder on its way to the socket:
		// Transcript, the first ReplyDelta and the first ReplyAudio are the
		// ASR/LLM/TTS boundaries, and Cancel closes the barge-in chain.
		send = rec.Wrap(send)

		var onTranscript func(asr.Transcript)
		if replyEnabled {
			onTranscript = func(t asr.Transcript) {
				if *verbose {
					log.Printf("[%s] TEXT 💬 %q", streamID, t.Text)
				}
				runner.Start(t)
			}
		}
		onUtt := makeOnUtterance(streamID, rec)

		// Create StreamWorker for streaming ASR
		var streamWorker *asr.StreamWorker
		if asrEnabled {
			swCfg := &asr.StreamWorkerConfig{
				GRPCAddress:   asrGRPCAddr,
				Model:         asrModel,
				Language:      asrLanguage,
				ChunkFrames:   asrChunkFrames,
				OverlapFrames: asrOverlapFrames,
				GraceFrames:   asrGraceFrames,
			}
			streamWorker = asr.NewStreamWorker(streamID, swCfg, func(t asr.StreamTranscript) {
				if strings.TrimSpace(t.Text) == "" {
					return
				}
				if *verbose {
					if t.IsFinal {
						log.Printf("[%s] TEXT 💬 (final) %q", streamID, t.Text)
					} else {
						log.Printf("[%s] TEXT 💬 (partial) %q", streamID, t.Text)
					}
				}
				// The ASR path owns transcript emission: partials stream as they
				// come, the final goes out once, and the reply runner only starts
				// the turn. runner is captured by reference; it is set before any
				// transcript can arrive.
				if err := send(&pb.ServerMessage{Msg: &pb.ServerMessage_Transcript{
					Transcript: &pb.Transcript{UtteranceId: t.StartSeq, Text: t.Text, IsFinal: t.IsFinal},
				}}); err != nil {
					return // client gone
				}
				if t.IsFinal && onTranscript != nil {
					onTranscript(asr.Transcript{Text: t.Text, StreamID: t.StreamID, StartSeq: t.StartSeq, EndSeq: t.EndSeq, FrameCount: t.FrameCount})
				}
			})
			if err := streamWorker.Start(); err != nil {
				log.Printf("stream %s: stream worker start FAILED (gRPC connection error): %v", streamID, err)
				streamWorker = nil
			} else {
				// grpc.NewClient is lazy: this says the worker is running, not that
				// the ASR service is reachable. A failure to reach it shows up per
				// utterance as "failed to create ASR stream".
				log.Printf("stream %s: asr worker running (target %s)", streamID, asrGRPCAddr)
			}
		}

		m := endpoint.NewMachine(streamID, epCfg, mode, onUtt, func() {
			rec.Barge()
			if runner == nil {
				return
			}
			if *verbose {
				log.Printf("[%s] BARG ✋ user talked over the reply — cancelling", streamID)
			}
			runner.Barge()
		}, func(startSeq uint64, preroll []byte) {
			if streamWorker == nil {
				return
			}
			if len(preroll) > 0 {
				// Barge-in: arm the ASR stream seeded with the buffered onset.
				streamWorker.BargeIn(startSeq, preroll)
			} else {
				streamWorker.Arm(startSeq)
			}
		}, func() {
			if streamWorker != nil {
				streamWorker.Disarm(0)
			}
		})
		if replyEnabled {
			runner = turn.NewRunner(streamID, send, llmClient, ttsClient, ttsClient.SampleRateHz(), m.SetSpeaking, lcfg.HistoryMaxTokens, lcfg.HistoryMaxTurns)
		}
		return m, streamWorker, runner
	}

	// Phase 8 timing taps. Same shape as the -verbose tracing below and applied
	// outside it, so a boundary is stamped before anything else reacts to it.
	// The frame indices these carry are retroactive — the VAD reports the frame
	// speech actually began on, not the one it worked that out on — so the
	// recorder resolves them through its own ring of frame arrival times.
	recVad := func(rec *metrics.Recorder, next func(vad.Event)) func(vad.Event) {
		return func(e vad.Event) {
			switch e.Type {
			case vad.EventStart:
				rec.VadStart(e.Frame)
			case vad.EventEnd:
				rec.VadEnd(e.Frame)
			}
			next(e)
		}
	}
	recWake := func(rec *metrics.Recorder, next func(wakeword.Event)) func(wakeword.Event) {
		return func(e wakeword.Event) {
			rec.Arm(e.Frame)
			next(e)
		}
	}
	recFrame := func(rec *metrics.Recorder, next func(uint64, []byte)) func(uint64, []byte) {
		return func(seq uint64, pcm []byte) {
			rec.Frame(seq)
			next(seq, pcm)
		}
	}

	// Under -verbose, tee the VAD/wake events to the log before the endpoint
	// machine consumes them (Phase 5 replaced the sinks' default loggers).
	traceVad := func(streamID string, next func(vad.Event)) func(vad.Event) {
		if !*verbose {
			return next
		}
		return func(e vad.Event) {
			switch e.Type {
			case vad.EventStart:
				log.Printf("[%s] VAD  ▶ SPEECH  (frame %d)", streamID, e.Frame)
			case vad.EventEnd:
				log.Printf("[%s] VAD  ■ silence (frame %d)", streamID, e.Frame)
			}
			next(e)
		}
	}
	traceWake := func(streamID string, next func(wakeword.Event)) func(wakeword.Event) {
		if !*verbose {
			return next
		}
		return func(e wakeword.Event) {
			log.Printf("[%s] WAKE 🔔 detected (frame %d) → listening", streamID, e.Frame)
			next(e)
		}
	}

	srv := server.New()
	if *wwModel == "" {
		srv.NewSink = func(streamID string, send server.Sender) server.Sink {
			rec := metrics.New(streamID, false)
			m, streamWorker, runner := buildTurn(streamID, send, endpoint.ArmOnVad, rec)
			frameForward := recFrame(rec, func(seq uint64, pcm []byte) {
				m.OnFrame(seq, pcm)
				if streamWorker != nil && m.IsArmed() {
					streamWorker.PushFrame(seq, pcm)
				}
			})
			return withDrain(vad.NewSink(streamID, engine, cfg,
				recVad(rec, traceVad(streamID, m.OnVad)), frameForward), streamWorker, runner)
		}
		log.Printf("astra listening on %s (vad: %s, wake word disabled)", *addr, *vadModel)
	} else {
		// Sidecar defaults to the model's own .json: the v1/v2 frontends differ
		// (mel bins, window length), so a stale -ww-config silently mis-scores.
		sidecar := *wwConfig
		if sidecar == "" {
			sidecar = strings.TrimSuffix(*wwModel, ".onnx") + ".json"
		}
		wwCfg, err := wakeword.LoadConfig(sidecar)
		if err != nil {
			log.Fatal(err)
		}
		wwEngine, err := wakeword.NewEngine(*wwModel, wwCfg)
		if err != nil {
			log.Fatal(err)
		}
		defer wwEngine.Close()
		srv.NewSink = func(streamID string, send server.Sender) server.Sink {
			rec := metrics.New(streamID, true)
			m, streamWorker, runner := buildTurn(streamID, send, endpoint.ArmOnWake, rec)
			frameForward := recFrame(rec, func(seq uint64, pcm []byte) {
				m.OnFrame(seq, pcm)
				if streamWorker != nil && m.IsArmed() {
					streamWorker.PushFrame(seq, pcm)
				}
			})
			return withDrain(wakeword.NewSink(streamID, engine, cfg, wwEngine, wwCfg,
				recVad(rec, traceVad(streamID, m.OnVad)),
				recWake(rec, traceWake(streamID, m.OnWake)),
				frameForward), streamWorker, runner)
		}
		log.Printf("astra listening on %s (vad: %s, wake word: %s)", *addr, *vadModel, *wwModel)
	}

	// The browser demo is served under /app/ so the WebSocket keeps the root
	// path the Python mic client already uses.
	mux := http.NewServeMux()
	mux.Handle("/", srv)
	if *webDir != "" {
		if _, err := os.Stat(*webDir); err != nil {
			log.Printf("web client disabled: %v", err)
		} else {
			mux.Handle("/app/", http.StripPrefix("/app/", http.FileServer(http.Dir(*webDir))))
			log.Printf("web client: http://localhost%s/app/", *addr)
		}
	}

	if err := http.ListenAndServe(*addr, mux); err != nil {
		log.Fatal(err)
	}
}
