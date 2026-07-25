package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/arnav/astra/services/backend/internal/endpoint"
	"github.com/arnav/astra/services/backend/internal/server"
	"github.com/arnav/astra/services/backend/internal/vad"
	"github.com/arnav/astra/services/backend/internal/wakeword"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	ortLib := flag.String("ort-lib", "", "path to onnxruntime shared library (default: $ASTRA_ORT_LIB, then well-known paths)")
	vadModel := flag.String("vad-model", "../../assets/models/vad/vad_v1.onnx", "path to VAD ONNX model (default assumes running from services/backend)")
	vadConfig := flag.String("vad-config", "../../assets/models/vad/vad_v1.json", "path to VAD sidecar config")
	wwModel := flag.String("ww-model", "", "path to merged wake-word ONNX model; empty disables wake word (VAD-only sink)")
	wwConfig := flag.String("ww-config", "../../assets/models/wakeword/ww_v1.json", "path to wake-word sidecar config")
	endpointConfig := flag.String("endpoint-config", "../../assets/configs/endpoint.json", "path to endpoint (utterance) config")
	endpointWavDir := flag.String("endpoint-wav-dir", "", "if set, dump each closed utterance here as WAV (debug/verification)")
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

	// onUtterance for a stream: WAV dumper when -endpoint-wav-dir is set, else
	// nil (the machine logs each utterance). Per-stream because the dumper owns
	// its own filename counter. Under -verbose, wrap it so the utterance end is
	// always logged even when a WAV dumper is active.
	makeOnUtterance := func(streamID string) func(endpoint.Utterance) {
		var next func(endpoint.Utterance)
		if *endpointWavDir != "" {
			dump, err := endpoint.NewWavDumper(*endpointWavDir, streamID)
			if err != nil {
				log.Printf("stream %s: wav dumper disabled: %v", streamID, err)
			} else {
				next = dump
			}
		}
		if !*verbose {
			return next
		}
		return func(u endpoint.Utterance) {
			log.Printf("[%s] UTTR ⏹ listening end — utterance %d frames (seq %d-%d)",
				streamID, u.FrameCount, u.StartSeq, u.EndSeq)
			if next != nil {
				next(u)
			}
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
		srv.NewSink = func(streamID string) server.Sink {
			m := endpoint.NewMachine(streamID, epCfg, endpoint.ArmOnVad, makeOnUtterance(streamID))
			return vad.NewSink(streamID, engine, cfg, traceVad(streamID, m.OnVad), m.OnFrame)
		}
		log.Printf("astra listening on %s (vad: %s, wake word disabled)", *addr, *vadModel)
	} else {
		wwCfg, err := wakeword.LoadConfig(*wwConfig)
		if err != nil {
			log.Fatal(err)
		}
		wwEngine, err := wakeword.NewEngine(*wwModel, wwCfg)
		if err != nil {
			log.Fatal(err)
		}
		defer wwEngine.Close()
		srv.NewSink = func(streamID string) server.Sink {
			m := endpoint.NewMachine(streamID, epCfg, endpoint.ArmOnWake, makeOnUtterance(streamID))
			return wakeword.NewSink(streamID, engine, cfg, wwEngine, wwCfg,
				traceVad(streamID, m.OnVad), traceWake(streamID, m.OnWake), m.OnFrame)
		}
		log.Printf("astra listening on %s (vad: %s, wake word: %s)", *addr, *vadModel, *wwModel)
	}

	if err := http.ListenAndServe(*addr, srv); err != nil {
		log.Fatal(err)
	}
}
