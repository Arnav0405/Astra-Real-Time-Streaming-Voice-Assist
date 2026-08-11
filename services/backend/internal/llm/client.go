package llm

import (
	"bufio"
	"bytes"
	"context"
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
}

// Client is an OpenAI-compatible streaming chat client. Safe for concurrent
// use. There is deliberately no retry: a retried turn would speak after the
// user has already moved on, and asr's retry exists only because a dropped
// utterance is unrecoverable.
type Client struct {
	cfg     Config
	key     string
	timeout time.Duration
}

// NewClient returns a client for cfg authenticating with apiKey.
func NewClient(cfg Config, apiKey string) *Client {
	return &Client{cfg: cfg, key: apiKey, timeout: requestTimeout}
}

type chatMessage struct {
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

	msgs := []chatMessage{}
	if c.cfg.SystemPrompt != "" {
		msgs = append(msgs, chatMessage{Role: "system", Content: c.cfg.SystemPrompt})
	}
	msgs = append(msgs, chatMessage{Role: "user", Content: prompt})

	body, err := json.Marshal(map[string]any{
		"model":      c.cfg.Model,
		"messages":   msgs,
		"max_tokens": c.cfg.MaxTokens,
		"stream":     true,
	})
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.key)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

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
