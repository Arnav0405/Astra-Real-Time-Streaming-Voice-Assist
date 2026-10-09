package llm

import (
	"context"
	"encoding/json"
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

// TestStreamSendsProviderIdentity is the OpenCode Go contract: requests
// without a stable x-opencode-session are rejected (400 MissingSessionID),
// and a named User-Agent is asked for rather than Go's library default.
func TestStreamSendsProviderIdentity(t *testing.T) {
	var gotSession, gotUA string
	var sessions []string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotSession = r.Header.Get("x-opencode-session")
		gotUA = r.Header.Get("User-Agent")
		sessions = append(sessions, gotSession)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer ts.Close()

	c := NewClient(testConfig(ts.URL), "key")
	for i := 0; i < 2; i++ {
		if err := c.Stream(context.Background(), "hi", func(string) {}); err != nil {
			t.Fatalf("Stream %d: %v", i, err)
		}
	}
	if gotSession == "" {
		t.Error("x-opencode-session missing; OpenCode Go rejects the request without it")
	}
	if sessions[0] != sessions[1] {
		t.Errorf("session id not stable across requests: %q vs %q", sessions[0], sessions[1])
	}
	if gotUA == "" || gotUA == "Go-http-client/1.1" {
		t.Errorf("User-Agent = %q, want a client-identifying value", gotUA)
	}
}

// TestStreamSendsReasoningEffort pins the body contract for reasoning models:
// mimo-v2.6-flash silently burns MaxTokens on reasoning_content unless
// reasoning_effort=none is sent (finish_reason=length, empty reply). When the
// config omits the field, the key must be absent so plain models don't see it.
func TestStreamSendsReasoningEffort(t *testing.T) {
	var gotBody map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// json.Decode merges into a non-nil map; clear first or the first
		// request's keys leak into later assertions.
		gotBody = nil
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer ts.Close()

	cfg := testConfig(ts.URL)
	cfg.ReasoningEffort = "none"
	c := NewClient(cfg, "key")
	if err := c.Stream(context.Background(), "hi", func(string) {}); err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if got, _ := gotBody["reasoning_effort"].(string); got != "none" {
		t.Errorf("reasoning_effort = %q, want \"none\" — without it the model reasons away the token budget", got)
	}

	cfg.ReasoningEffort = ""
	c = NewClient(cfg, "key")
	if err := c.Stream(context.Background(), "hi", func(string) {}); err != nil {
		t.Fatalf("Stream (unset): %v", err)
	}
	if _, ok := gotBody["reasoning_effort"]; ok {
		t.Errorf("reasoning_effort present with empty config: %v", gotBody["reasoning_effort"])
	}
}
