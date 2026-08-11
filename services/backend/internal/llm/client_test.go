package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func testConfig(baseURL string) Config {
	return Config{BaseURL: baseURL, Model: "test-model", SystemPrompt: "be brief", MaxTokens: 128}
}

// sseServer streams the given payload lines, flushing after each so the
// client sees them as they arrive rather than as one buffered body.
func sseServer(t *testing.T, lines []string, gap time.Duration) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, ok := w.(http.Flusher)
		if !ok {
			t.Error("test server response writer is not a Flusher")
			return
		}
		for _, l := range lines {
			fmt.Fprintf(w, "%s\n\n", l)
			fl.Flush()
			if gap > 0 {
				time.Sleep(gap)
			}
		}
	}))
}

func delta(s string) string {
	return fmt.Sprintf(`data: {"choices":[{"delta":{"content":%q}}]}`, s)
}

func TestStreamCollectsDeltasInOrder(t *testing.T) {
	ts := sseServer(t, []string{delta("Hello"), delta(", "), delta("world"), "data: [DONE]"}, 0)
	defer ts.Close()

	var got strings.Builder
	err := NewClient(testConfig(ts.URL), "key").Stream(context.Background(), "hi", func(s string) {
		got.WriteString(s)
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if got.String() != "Hello, world" {
		t.Errorf("got %q, want %q", got.String(), "Hello, world")
	}
}

func TestStreamSkipsMalformedChunks(t *testing.T) {
	// One garbage event must not silence the rest of an arriving reply.
	ts := sseServer(t, []string{delta("ok "), "data: {not json", ": keepalive", delta("still here"), "data: [DONE]"}, 0)
	defer ts.Close()

	var got strings.Builder
	if err := NewClient(testConfig(ts.URL), "key").Stream(context.Background(), "hi", func(s string) {
		got.WriteString(s)
	}); err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if got.String() != "ok still here" {
		t.Errorf("got %q, want %q", got.String(), "ok still here")
	}
}

// TestStreamCancelAbortsRequest is the barge-in guarantee: cancelling must
// tear down the HTTP request so the provider stops generating, not merely
// stop us from reading. Asserting only that Stream returns would pass even if
// the body kept streaming in the background.
func TestStreamCancelAbortsRequest(t *testing.T) {
	disconnected := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		fmt.Fprintf(w, "%s\n\n", delta("first"))
		fl.Flush()
		select {
		case <-r.Context().Done(): // the client actually hung up
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
		done <- NewClient(testConfig(ts.URL), "key").Stream(ctx, "hi", func(string) { cancel() })
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("Stream returned nil error after cancellation")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Stream did not return after cancellation")
	}

	select {
	case <-disconnected:
	case <-time.After(2 * time.Second):
		t.Fatal("request was not aborted; provider would keep generating")
	}
}

func TestStreamHTTPErrorIsReported(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "rate limited", http.StatusTooManyRequests)
	}))
	defer ts.Close()

	err := NewClient(testConfig(ts.URL), "key").Stream(context.Background(), "hi", func(string) {})
	if err == nil || !strings.Contains(err.Error(), "429") {
		t.Fatalf("want an error mentioning 429, got %v", err)
	}
}

func TestStreamSendsModelAndAuth(t *testing.T) {
	var gotAuth, gotBody string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		b := make([]byte, r.ContentLength)
		r.Body.Read(b)
		gotBody = string(b)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer ts.Close()

	if err := NewClient(testConfig(ts.URL), "secret").Stream(context.Background(), "hi", func(string) {}); err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if gotAuth != "Bearer secret" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	for _, want := range []string{`"model":"test-model"`, `"stream":true`, "be brief"} {
		if !strings.Contains(gotBody, want) {
			t.Errorf("request body missing %s: %s", want, gotBody)
		}
	}
}
