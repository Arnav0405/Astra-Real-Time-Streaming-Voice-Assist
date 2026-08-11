// Capture worklet: converts the mic stream into the exact Frame the server
// accepts — 320 samples of 16 kHz mono s16le, i.e. 20 ms.
//
// There is no resampling here on purpose. The page creates its capture
// AudioContext at 16 kHz, so the browser's own media pipeline does the
// conversion before this processor ever sees a sample.

const FRAME_SAMPLES = 320; // 20 ms at 16 kHz

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Int16Array(FRAME_SAMPLES);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true; // mic not delivering yet; stay alive

    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      // Asymmetric scaling: s16le runs -32768..32767, so a full-scale
      // negative sample would clip if it shared the positive factor.
      this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.n === FRAME_SAMPLES) {
        const frame = this.buf.slice(); // buf is reused; hand over a copy
        this.port.postMessage(frame, [frame.buffer]);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor('capture', CaptureProcessor);
