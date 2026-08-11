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
	"github.com/arnav/astra/services/backend/internal/server"
	"github.com/arnav/astra/services/backend/internal/vad"
	"github.com/arnav/astra/services/backend/internal/wakeword"
)

// drainSink lets the ASR worker finish queued/in-flight transcriptions after
// the stream's frames are drained. Close runs in the background so socket
// teardown never waits on a slow transcription HTTP call.
type drainSink struct {
	server.Sink
	worker *asr.Worker
}

func (s drainSink) Run(frames <-chan server.Frame) {
	s.Sink.Run(frames)
	go s.worker.Close()
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
	envFile := flag.String("env-file", "../../.env", "path to .env file with NAGA_API_KEY (already-exported env wins)")
	verbose := flag.Bool("verbose", false, "log the per-stream pipeline trace (VAD speech/silence, wake, utterance)")
	flag.Parse()

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

	// ASR is mandatory unless explicitly disabled: a missing key must be a
	// choice you typed (-no-asr), never a silent fallback.
	var asrClient *asr.Client
	if *noASR {
		log.Print("asr disabled (-no-asr): utterances will not be transcribed")
	} else {
		godotenv.Load(*envFile) // best effort; exported env wins over the file
		key := os.Getenv("NAGA_API_KEY")
		if key == "" {
			log.Fatalf("NAGA_API_KEY not set (checked environment and %s); pass -no-asr to run without transcription", *envFile)
		}
		acfg, err := asr.LoadConfig(*asrConfig)
		if err != nil {
			log.Fatal(err)
		}
		asrClient = asr.NewClient(acfg, key)
		log.Printf("asr enabled: %s model %s", acfg.BaseURL, acfg.Model)
	}

	// onUtterance for a stream: optional WAV dumper (-endpoint-wav-dir), then
	// the ASR worker; with neither, nil (the machine logs each utterance).
	// Per-stream because the dumper owns its filename counter and the worker
	// its queue. Under -verbose, wrap it so the utterance end is always logged.
	// The returned worker is nil when ASR is disabled.
	makeOnUtterance := func(streamID string) (func(endpoint.Utterance), *asr.Worker) {
		var next func(endpoint.Utterance)
		var worker *asr.Worker
		if asrClient != nil {
			// nil onTranscript = server-side transcript log; Phase 7's LLM
			// consumer plugs in here.
			worker = asr.NewWorker(streamID, asrClient, nil)
			next = func(u endpoint.Utterance) { worker.Enqueue(u) }
		}
		if *endpointWavDir != "" {
			dump, err := endpoint.NewWavDumper(*endpointWavDir, streamID)
			if err != nil {
				log.Printf("stream %s: wav dumper disabled: %v", streamID, err)
			} else {
				enqueue := next
				next = func(u endpoint.Utterance) {
					dump(u)
					if enqueue != nil {
						enqueue(u)
					}
				}
			}
		}
		if !*verbose {
			return next, worker
		}
		return func(u endpoint.Utterance) {
			log.Printf("[%s] UTTR ⏹ listening end — utterance %d frames (seq %d-%d)",
				streamID, u.FrameCount, u.StartSeq, u.EndSeq)
			if next != nil {
				next(u)
			}
		}, worker
	}

	// withDrain wraps a sink so the stream's ASR worker drains after the
	// frame channel closes (Q10: last words still transcribe).
	withDrain := func(s server.Sink, w *asr.Worker) server.Sink {
		if w == nil {
			return s
		}
		return drainSink{Sink: s, worker: w}
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
		srv.NewSink = func(streamID string) server.Sink {
			onUtt, worker := makeOnUtterance(streamID)
			m := endpoint.NewMachine(streamID, epCfg, endpoint.ArmOnVad, onUtt)
			return withDrain(vad.NewSink(streamID, engine, cfg, traceVad(streamID, m.OnVad), m.OnFrame), worker)
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
		srv.NewSink = func(streamID string) server.Sink {
			onUtt, worker := makeOnUtterance(streamID)
			m := endpoint.NewMachine(streamID, epCfg, endpoint.ArmOnWake, onUtt)
			return withDrain(wakeword.NewSink(streamID, engine, cfg, wwEngine, wwCfg,
				traceVad(streamID, m.OnVad), traceWake(streamID, m.OnWake), m.OnFrame), worker)
		}
		log.Printf("astra listening on %s (vad: %s, wake word: %s)", *addr, *vadModel, *wwModel)
	}

	if err := http.ListenAndServe(*addr, srv); err != nil {
		log.Fatal(err)
	}
}
