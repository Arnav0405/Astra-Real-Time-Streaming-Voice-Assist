package tts

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	requestTimeout = 30 * time.Second
	// chunkBytes is how much PCM is forwarded per ReplyAudio message. It only
	// sets network granularity — barge-in responsiveness comes from the
	// client-side buffer flush, not from chunk size — so this is picked to
	// keep message overhead low, not to be small.
	chunkBytes = 4096
)

// Synthesizer turns one span of reply text into streamed PCM. *Client
// implements it; tests substitute a fake.
type Synthesizer interface {
	Speak(ctx context.Context, text string, onPCM func([]byte) error) error
}

// Client is an OpenAI-compatible streaming speech client. Safe for concurrent
// use. No retry, for the same reason as llm: a retried sentence arrives after
// the conversation has moved on.
type Client struct {
	cfg     Config
	key     string
	timeout time.Duration
}

// NewClient returns a client for cfg authenticating with apiKey.
func NewClient(cfg Config, apiKey string) *Client {
	return &Client{cfg: cfg, key: apiKey, timeout: requestTimeout}
}

// SampleRateHz is the rate of the PCM Speak produces.
func (c *Client) SampleRateHz() int { return c.cfg.SampleRateHz }

// Speak synthesizes text and hands PCM to onPCM in arrival order, from the
// calling goroutine. The buffer passed to onPCM is reused after it returns,
// so a consumer that keeps it must copy. Returns early if onPCM errors (the
// client is gone) or ctx is cancelled (barge-in).
func (c *Client) Speak(ctx context.Context, text string, onPCM func([]byte) error) error {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	body, err := json.Marshal(map[string]any{
		"model":           c.cfg.Model,
		"voice":           c.cfg.Voice,
		"input":           text,
		"response_format": c.cfg.Format,
	})
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/audio/speech", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.key)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("tts http %d: %s", resp.StatusCode, bytes.TrimSpace(b))
	}

	// s16le: a chunk must never end mid-sample or the client reassembles noise
	// at the seam. A read that lands on an odd boundary carries its trailing
	// byte into the next chunk — dropping it instead would byte-shift, and so
	// destroy, every sample that follows.
	buf := make([]byte, chunkBytes)
	carry := 0
	for {
		n, err := resp.Body.Read(buf[carry:])
		total := carry + n
		emit := total &^ 1 // round down to a whole number of samples
		if emit > 0 {
			if cbErr := onPCM(buf[:emit]); cbErr != nil {
				return cbErr
			}
		}
		carry = total - emit
		if carry == 1 {
			buf[0] = buf[emit]
		}
		if err == io.EOF {
			return nil // a lone trailing byte means a truncated stream; drop it
		}
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return ctxErr
			}
			return err
		}
	}
}
