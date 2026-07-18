package main

import (
	"flag"
	"log"
	"net/http"

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

	srv := server.New()
	if *wwModel == "" {
		srv.NewSink = func(streamID string) server.Sink {
			return vad.NewSink(streamID, engine, cfg, nil)
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
			return wakeword.NewSink(streamID, engine, cfg, wwEngine, wwCfg, nil, nil)
		}
		log.Printf("astra listening on %s (vad: %s, wake word: %s)", *addr, *vadModel, *wwModel)
	}

	if err := http.ListenAndServe(*addr, srv); err != nil {
		log.Fatal(err)
	}
}
