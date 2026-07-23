package endpoint

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"path/filepath"
)

// Fixed capture format (matches the strict WebSocket ingest contract).
const (
	wavSampleRate    = 16000
	wavChannels      = 1
	wavBitsPerSample = 16
)

// NewWavDumper returns an onUtterance handler that writes each closed utterance
// to <dir>/<streamID>_NNN.wav for playback verification. Debug output — best
// effort, errors are logged not fatal.
func NewWavDumper(dir, streamID string) (func(Utterance), error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("endpoint wav dir: %w", err)
	}
	n := 0
	return func(u Utterance) {
		path := filepath.Join(dir, fmt.Sprintf("%s_%03d.wav", streamID, n))
		n++
		if err := writeWav(path, u.PCM); err != nil {
			log.Printf("stream %s: wav dump %s: %v", streamID, path, err)
			return
		}
		log.Printf("stream %s: utterance -> %s (%d frames, seq %d-%d)",
			streamID, path, u.FrameCount, u.StartSeq, u.EndSeq)
	}, nil
}

// writeWav writes a canonical 44-byte-header PCM WAV. ponytail: no wav lib —
// the header is 11 fixed fields, one os.Create.
func writeWav(path string, pcm []byte) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	dataLen := uint32(len(pcm))
	byteRate := uint32(wavSampleRate * wavChannels * wavBitsPerSample / 8)
	blockAlign := uint16(wavChannels * wavBitsPerSample / 8)

	w.WriteString("RIFF")
	binary.Write(w, binary.LittleEndian, 36+dataLen) // chunk size
	w.WriteString("WAVE")
	w.WriteString("fmt ")
	binary.Write(w, binary.LittleEndian, uint32(16)) // subchunk1 size
	binary.Write(w, binary.LittleEndian, uint16(1))  // PCM
	binary.Write(w, binary.LittleEndian, uint16(wavChannels))
	binary.Write(w, binary.LittleEndian, uint32(wavSampleRate))
	binary.Write(w, binary.LittleEndian, byteRate)
	binary.Write(w, binary.LittleEndian, blockAlign)
	binary.Write(w, binary.LittleEndian, uint16(wavBitsPerSample))
	w.WriteString("data")
	binary.Write(w, binary.LittleEndian, dataLen)
	w.Write(pcm)
	return w.Flush()
}
