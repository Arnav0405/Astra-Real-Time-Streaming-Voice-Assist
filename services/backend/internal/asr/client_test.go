package asr

import (
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testConfig(baseURL string) Config {
	return Config{BaseURL: baseURL, Model: "whisper-large-v3:free", Language: "en"}
}

// fastClient returns a client with timings shrunk so failure paths don't
// stall the test run.
func fastClient(baseURL string) *Client {
	c := NewClient(testConfig(baseURL), "sekret")
	c.timeout = 200 * time.Millisecond
	c.backoff = time.Millisecond
	return c
}

func TestTranscribeRequestShape(t *testing.T) {
	pcm := []byte{1, 2, 3, 4, 5, 6}
	var gotPath, gotAuth, gotModel, gotLang, gotName string
	var gotFile []byte

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("parse multipart: %v", err)
		}
		gotModel = r.FormValue("model")
		gotLang = r.FormValue("language")
		f, hdr, err := r.FormFile("file")
		if err != nil {
			t.Errorf("form file: %v", err)
		} else {
			gotName = hdr.Filename
			gotFile, _ = io.ReadAll(f)
			f.Close()
		}
		fmt.Fprint(w, `{"text":"hello world"}`)
	}))
	defer srv.Close()

	text, err := fastClient(srv.URL).Transcribe(pcm)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if text != "hello world" {
		t.Errorf("text = %q, want %q", text, "hello world")
	}
	if gotPath != "/audio/transcriptions" {
		t.Errorf("path = %q", gotPath)
	}
	if gotAuth != "Bearer sekret" {
		t.Errorf("auth = %q", gotAuth)
	}
	if gotModel != "whisper-large-v3:free" {
		t.Errorf("model = %q", gotModel)
	}
	if gotLang != "en" {
		t.Errorf("language = %q", gotLang)
	}
	if !strings.HasSuffix(gotName, ".wav") {
		t.Errorf("filename = %q, want *.wav", gotName)
	}

	// WAV envelope: 44-byte canonical header + raw PCM.
	if len(gotFile) != 44+len(pcm) {
		t.Fatalf("file len = %d, want %d", len(gotFile), 44+len(pcm))
	}
	if string(gotFile[:4]) != "RIFF" || string(gotFile[8:12]) != "WAVE" {
		t.Errorf("bad wav magic: % x", gotFile[:12])
	}
	if rate := binary.LittleEndian.Uint32(gotFile[24:28]); rate != 16000 {
		t.Errorf("sample rate = %d, want 16000", rate)
	}
	if dataLen := binary.LittleEndian.Uint32(gotFile[40:44]); dataLen != uint32(len(pcm)) {
		t.Errorf("data len = %d, want %d", dataLen, len(pcm))
	}
	if string(gotFile[44:]) != string(pcm) {
		t.Errorf("pcm payload mismatch")
	}
}

func TestTranscribeOmitsEmptyLanguage(t *testing.T) {
	var hasLang bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.ParseMultipartForm(1 << 20)
		_, hasLang = r.MultipartForm.Value["language"]
		fmt.Fprint(w, `{"text":"x"}`)
	}))
	defer srv.Close()

	c := NewClient(Config{BaseURL: srv.URL, Model: "m"}, "k")
	if _, err := c.Transcribe([]byte{0, 0}); err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if hasLang {
		t.Error("language field sent despite empty config")
	}
}

func TestTranscribeRetriesOnceThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		fmt.Fprint(w, `{"text":"second try"}`)
	}))
	defer srv.Close()

	text, err := fastClient(srv.URL).Transcribe([]byte{0, 0})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if text != "second try" {
		t.Errorf("text = %q", text)
	}
	if n := calls.Load(); n != 2 {
		t.Errorf("calls = %d, want 2", n)
	}
}

func TestTranscribeGivesUpAfterRetry(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	_, err := fastClient(srv.URL).Transcribe([]byte{0, 0})
	if err == nil {
		t.Fatal("want error, got nil")
	}
	if n := calls.Load(); n != 2 {
		t.Errorf("calls = %d, want 2 (1 try + 1 retry)", n)
	}
}

func TestTranscribeTimesOut(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(500 * time.Millisecond)
		fmt.Fprint(w, `{"text":"too late"}`)
	}))
	defer srv.Close()

	c := fastClient(srv.URL)
	c.timeout = 20 * time.Millisecond
	start := time.Now()
	_, err := c.Transcribe([]byte{0, 0})
	if err == nil {
		t.Fatal("want timeout error, got nil")
	}
	if elapsed := time.Since(start); elapsed > 300*time.Millisecond {
		t.Errorf("took %v, timeout not enforced", elapsed)
	}
}
