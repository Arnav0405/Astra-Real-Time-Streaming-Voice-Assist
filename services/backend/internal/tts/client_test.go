package tts

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func testConfig(baseURL string) Config {
	return Config{BaseURL: baseURL, Model: "test-tts", Voice: "alloy", Format: "pcm", SampleRateHz: 24000}
}

// collect runs Speak against a server that writes body in odd-sized flushes,
// which is what forces the carry logic to be exercised.
func collect(t *testing.T, body []byte, flush int) []byte {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fl := w.(http.Flusher)
		for i := 0; i < len(body); i += flush {
			end := min(i+flush, len(body))
			w.Write(body[i:end])
			fl.Flush()
		}
	}))
	defer ts.Close()

	var got bytes.Buffer
	err := NewClient(testConfig(ts.URL), "key").Speak(context.Background(), "hello", func(p []byte) error {
		if len(p)%2 != 0 {
			t.Errorf("chunk of %d bytes splits an s16le sample", len(p))
		}
		got.Write(p) // p is reused; Write copies
		return nil
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	return got.Bytes()
}

// TestSpeakPreservesBytesAcrossOddFlushes is the regression guard for sample
// alignment: an odd-sized read must carry its trailing byte forward, not drop
// it. Dropping byte-shifts everything after and turns speech into static.
func TestSpeakPreservesBytesAcrossOddFlushes(t *testing.T) {
	body := make([]byte, 10000)
	for i := range body {
		body[i] = byte(i % 251)
	}
	for _, flush := range []int{1, 3, 4095, 4097, 9999} {
		got := collect(t, body, flush)
		if !bytes.Equal(got, body) {
			t.Errorf("flush=%d: got %d bytes, want %d, equal=%v", flush, len(got), len(body), bytes.Equal(got, body))
		}
	}
}

func TestSpeakStopsWhenConsumerErrors(t *testing.T) {
	body := make([]byte, 100000)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(body)
	}))
	defer ts.Close()

	sentinel := errors.New("client gone")
	calls := 0
	err := NewClient(testConfig(ts.URL), "key").Speak(context.Background(), "hi", func([]byte) error {
		calls++
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("want sentinel error, got %v", err)
	}
	if calls != 1 {
		t.Errorf("consumer called %d times after erroring, want 1", calls)
	}
}

func TestSpeakCancelAbortsRequest(t *testing.T) {
	disconnected := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fl := w.(http.Flusher)
		w.Write(make([]byte, chunkBytes))
		fl.Flush()
		select {
		case <-r.Context().Done():
			close(disconnected)
		case <-time.After(5 * time.Second):
			t.Error("server never saw the client disconnect")
		}
	}))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- NewClient(testConfig(ts.URL), "key").Speak(ctx, "hi", func([]byte) error {
			cancel()
			return nil
		})
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Speak did not return after cancellation")
	}
	select {
	case <-disconnected:
	case <-time.After(2 * time.Second):
		t.Fatal("request was not aborted; provider would keep synthesizing")
	}
}

func TestSpeakSendsVoiceAndFormat(t *testing.T) {
	var gotBody, gotAuth string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		b := make([]byte, r.ContentLength)
		r.Body.Read(b)
		gotBody = string(b)
	}))
	defer ts.Close()

	if err := NewClient(testConfig(ts.URL), "secret").Speak(context.Background(), "say this", func([]byte) error { return nil }); err != nil {
		t.Fatalf("Speak: %v", err)
	}
	if gotAuth != "Bearer secret" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	for _, want := range []string{`"voice":"alloy"`, `"response_format":"pcm"`, `"input":"say this"`} {
		if !strings.Contains(gotBody, want) {
			t.Errorf("request body missing %s: %s", want, gotBody)
		}
	}
}

func TestSpeakHTTPErrorIsReported(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusBadRequest)
	}))
	defer ts.Close()

	err := NewClient(testConfig(ts.URL), "key").Speak(context.Background(), "hi", func([]byte) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "400") {
		t.Fatalf("want an error mentioning 400, got %v", err)
	}
}
