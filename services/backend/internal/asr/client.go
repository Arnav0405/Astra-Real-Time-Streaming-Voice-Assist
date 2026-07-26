package asr

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/arnav/astra/services/backend/internal/endpoint"
)

const (
	defaultTimeout = 30 * time.Second
	defaultBackoff = 500 * time.Millisecond
	maxAttempts    = 2 // 1 try + 1 retry; more just poisons queue latency
)

// Client posts one utterance's PCM to the transcription endpoint as an
// in-memory WAV. Safe for concurrent use.
type Client struct {
	cfg     Config
	key     string
	timeout time.Duration
	backoff time.Duration
}

// NewClient returns a client for cfg authenticating with apiKey.
func NewClient(cfg Config, apiKey string) *Client {
	return &Client{cfg: cfg, key: apiKey, timeout: defaultTimeout, backoff: defaultBackoff}
}

// Transcribe uploads pcm (16 kHz mono s16le) and returns the transcript text.
// One retry after a short backoff, then the error is the caller's to drop.
func (c *Client) Transcribe(pcm []byte) (string, error) {
	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			time.Sleep(c.backoff)
		}
		text, err := c.once(pcm)
		if err == nil {
			return text, nil
		}
		lastErr = err
	}
	return "", lastErr
}

func (c *Client) once(pcm []byte) (string, error) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, err := mw.CreateFormFile("file", "utterance.wav")
	if err != nil {
		return "", err
	}
	fw.Write(endpoint.WavBytes(pcm))
	mw.WriteField("model", c.cfg.Model)
	if c.cfg.Language != "" {
		mw.WriteField("language", c.cfg.Language)
	}
	mw.Close()

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/audio/transcriptions", &body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+c.key)
	req.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("asr http %d: %s", resp.StatusCode, bytes.TrimSpace(b))
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("asr response decode: %w", err)
	}
	return out.Text, nil
}
