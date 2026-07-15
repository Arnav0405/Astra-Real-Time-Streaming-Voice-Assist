package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/arnav/astra/services/backend/internal/server"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	flag.Parse()

	srv := server.New()
	log.Printf("astra listening on %s", *addr)
	if err := http.ListenAndServe(*addr, srv); err != nil {
		log.Fatal(err)
	}
}
