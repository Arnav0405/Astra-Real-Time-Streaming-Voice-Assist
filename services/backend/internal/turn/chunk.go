package turn

import (
	"strings"
	"unicode/utf8"
)

const (
	// minChunkBytes stops a chunk breaking at punctuation too early to be a
	// real sentence. It is what keeps "Dr. Smith" or "No. 4" from being
	// synthesized as a two-word utterance of its own.
	minChunkBytes = 12
	// maxChunkBytes force-flushes a reply that never punctuates, so speech
	// still starts instead of waiting for the whole generation.
	maxChunkBytes = 240
)

// nextChunk splits the leading speakable span off buf. ok is false when buf
// holds no complete span yet and the caller should wait for more text.
//
// Splitting happens on terminal punctuation followed by whitespace: requiring
// the whitespace is what keeps decimals like "3.14" intact, since the period
// there is followed by a digit.
func nextChunk(buf string) (chunk, rest string, ok bool) {
	for i := 0; i < len(buf); i++ {
		switch buf[i] {
		case '.', '!', '?', '\n':
		default:
			continue
		}
		end := i + 1
		// A newline is itself the break, so it needs no lookahead. The other
		// marks do: the whitespace after them is what distinguishes the end of
		// a sentence from the dot in "3.14".
		if buf[i] != '\n' {
			if end >= len(buf) {
				// Can't tell yet whether whitespace follows; wait for more
				// text. Nothing later in buf can be a boundary either, since
				// this is the last byte.
				return "", buf, false
			}
			if !isSpace(buf[end]) {
				continue
			}
		}
		if end < minChunkBytes {
			continue
		}
		for end < len(buf) && isSpace(buf[end]) {
			end++ // trailing whitespace rides along rather than starting the next chunk
		}
		return buf[:end], buf[end:], true
	}

	if len(buf) >= maxChunkBytes {
		cut := strings.LastIndexByte(buf[:maxChunkBytes], ' ')
		if cut <= 0 {
			// No word boundary at all — back up to a rune boundary so the
			// split never lands inside a multi-byte character.
			cut = maxChunkBytes
			for cut > 0 && !utf8.RuneStart(buf[cut]) {
				cut--
			}
		}
		if cut > 0 {
			return buf[:cut], buf[cut:], true
		}
	}
	return "", buf, false
}

func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}
