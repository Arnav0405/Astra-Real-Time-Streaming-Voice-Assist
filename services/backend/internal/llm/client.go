package llm

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// requestTimeout bounds a whole reply. It is generous because the turn is
// normally ended by ctx cancellation (barge-in) or by the provider, not here.
const requestTimeout = 60 * time.Second

// Streamer produces an assistant reply for one transcript, calling onDelta as
// text arrives. *Client implements it; tests substitute a fake.
type Streamer interface {
	Stream(ctx context.Context, prompt string, onDelta func(string)) error
	StreamWithHistory(ctx context.Context, prompt string, history []ChatMessage, onDelta func(string)) error
}

// Client is an OpenAI-compatible streaming chat client. Safe for concurrent
// use. There is deliberately no retry: a retried turn would speak after the
// user has already moved on, and asr's retry exists only because a dropped
// utterance is unrecoverable.
type Client struct {
	cfg     Config
	key     string
	timeout time.Duration
	// session is the stable x-opencode-session id sent with every request.
	// OpenCode Go rejects requests missing it (400 MissingSessionID) and uses
	// it for routing and prompt caching, so it must not change mid-run.
	session string
}

// userAgent identifies the client. OpenCode Go asks callers to name their
// client rather than send a generic HTTP-library default (Go's would be
// "Go-http-client/1.1"); other providers ignore the header.
const userAgent = "astra-voice/1.0"

// NewClient returns a client for cfg authenticating with apiKey.
func NewClient(cfg Config, apiKey string) *Client {
	return &Client{cfg: cfg, key: apiKey, timeout: requestTimeout, session: newSessionID()}
}

// newSessionID returns a random id stable for the lifetime of the process.
func newSessionID() string {
	var b [16]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Stream sends prompt and invokes onDelta for each chunk of reply text, in
// order, from the calling goroutine. It returns when the reply completes, the
// provider errors, or ctx is cancelled — cancelling aborts the HTTP body read
// so the provider stops generating.
func (c *Client) Stream(ctx context.Context, prompt string, onDelta func(string)) error {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	msgs := []ChatMessage{}
	if c.cfg.SystemPrompt != "" {
		msgs = append(msgs, ChatMessage{Role: "system", Content: c.cfg.SystemPrompt})
	}
	msgs = append(msgs, ChatMessage{Role: "user", Content: prompt})

	body, err := c.requestBody(msgs)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	c.setHeaders(req)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("llm http %d: %s", resp.StatusCode, bytes.TrimSpace(b))
	}
	return parseSSE(ctx, resp.Body, onDelta)
}

// setHeaders authenticates and identifies a request: the bearer key, the
// SSE accept, plus OpenCode Go's required session id and a named user agent.
func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Authorization", "Bearer "+c.key)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("x-opencode-session", c.session)
	req.Header.Set("User-Agent", userAgent)
}

// requestBody builds the JSON body shared by both stream paths.
// ReasoningEffort is included only when configured: mimo-v2.6-flash reasons
// by default and burns MaxTokens on invisible reasoning_content deltas
// (finish_reason=length, empty reply), so "none" is what makes the configured
// token budget mean spoken tokens.
func (c *Client) requestBody(msgs []ChatMessage) ([]byte, error) {
	payload := map[string]any{
		"model":      c.cfg.Model,
		"messages":   msgs,
		"max_tokens": c.cfg.MaxTokens,
		"stream":     true,
	}
	if c.cfg.ReasoningEffort != "" {
		payload["reasoning_effort"] = c.cfg.ReasoningEffort
	}
	return json.Marshal(payload)
}

// StreamWithHistory sends prompt with conversation history and invokes onDelta
// for each chunk of reply text. History is truncated to fit within
// HistoryMaxTokens and HistoryMaxTurns.
func (c *Client) StreamWithHistory(ctx context.Context, prompt string, history []ChatMessage, onDelta func(string)) error {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Build messages: system prompt + truncated history + current prompt
	msgs := []ChatMessage{}
	if c.cfg.SystemPrompt != "" {
		msgs = append(msgs, ChatMessage{Role: "system", Content: c.cfg.SystemPrompt})
	}

	// Truncate history from oldest to fit token/turn limits
	msgs = append(msgs, truncateHistory(history, c.cfg.HistoryMaxTokens, c.cfg.HistoryMaxTurns)...)
	msgs = append(msgs, ChatMessage{Role: "user", Content: prompt})

	body, err := c.requestBody(msgs)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	c.setHeaders(req)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("llm http %d: %s", resp.StatusCode, bytes.TrimSpace(b))
	}
	return parseSSE(ctx, resp.Body, onDelta)
}

// truncateHistory truncates conversation history from the oldest messages
// to fit within maxTokens and maxTurns. Each turn is a user+assistant pair.
// Rough token estimation: ~4 chars per token for English text.
func truncateHistory(history []ChatMessage, maxTokens, maxTurns int) []ChatMessage {
	if len(history) == 0 {
		return nil
	}

	// Count turns (user+assistant pairs)
	turns := 0
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "user" {
			turns++
			if turns >= maxTurns {
				// Keep from this index onwards
				history = history[i:]
				break
			}
		}
	}

	// Estimate tokens and truncate if needed
	// Simple estimation: 4 chars = 1 token
	estimatedTokens := 0
	for _, msg := range history {
		estimatedTokens += len(msg.Content) / 4
	}

	// If over token budget, drop oldest messages (which are at the start after turn truncation)
	// But keep at least the most recent turn
	for estimatedTokens > maxTokens && len(history) > 2 {
		// Remove oldest pair (assuming user+assistant pairs)
		history = history[2:]
		estimatedTokens = 0
		for _, msg := range history {
			estimatedTokens += len(msg.Content) / 4
		}
	}

	return history
}

// parseSSE consumes an OpenAI-style `data:` event stream. Unparseable events
// are skipped rather than fatal: one malformed chunk should not silence a
// reply that is otherwise arriving fine.
func parseSSE(ctx context.Context, r io.Reader, onDelta func(string)) error {
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		// Read errors are the normal cancellation path, but a provider that
		// pauses mid-stream would otherwise block here past the cancel.
		if err := ctx.Err(); err != nil {
			return err
		}
		line := strings.TrimSpace(sc.Text())
		payload, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue // blank keepalive line or a field we don't use
		}
		payload = strings.TrimSpace(payload)
		if payload == "[DONE]" {
			return nil
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		for _, ch := range chunk.Choices {
			if ch.Delta.Content != "" {
				onDelta(ch.Delta.Content)
			}
		}
	}
	if err := sc.Err(); err != nil {
		// A cancelled context surfaces here as a transport error; report the
		// cause so the caller can tell barge-in from a real failure.
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		return err
	}
	return nil
}
