// Astra browser client.
//
// Why the browser at all: barge-in needs the microphone live while the
// assistant is speaking, which means the mic hears the assistant. Rather than
// write an acoustic echo canceller, this uses the one the browser already
// ships — getUserMedia({echoCancellation:true}) is WebRTC's AEC3.

const FRAME_MS = 20;
const CAPTURE_RATE = 16000;

const els = {
  toggle: document.getElementById('toggle'),
  status: document.getElementById('status'),
  dot: document.getElementById('dot'),
  transcript: document.getElementById('transcript'),
  reply: document.getElementById('reply'),
  log: document.getElementById('log'),
};

let ClientMessage, ServerMessage;
let ws = null;
let captureCtx = null;
let micStream = null;
let seq = 0;
let running = false;

// --- Playback -------------------------------------------------------------
//
// Reply audio is scheduled as a chain of AudioBufferSourceNodes rather than
// pushed through a ring-buffer worklet. Sequential scheduling is what the Web
// Audio API is for, and it makes the flush trivial: stopping every scheduled
// source IS the instant cut-off that barge-in needs.

let playCtx = null;
let nextStart = 0;
let sources = [];

function playChunk(pcm, rate) {
  if (!playCtx || playCtx.sampleRate !== rate) {
    if (playCtx) playCtx.close();
    // The context is created at the provider's rate, so the browser resamples
    // to the output device natively and the server never has to.
    playCtx = new AudioContext({ sampleRate: rate });
    nextStart = 0;
  }

  // Copy before viewing as Int16: the decoded bytes are a view into a larger
  // buffer at an arbitrary offset, and Int16Array needs 2-byte alignment.
  const bytes = new Uint8Array(pcm);
  const samples = new Int16Array(bytes.buffer);

  const buf = playCtx.createBuffer(1, samples.length, rate);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < samples.length; i++) ch[i] = samples[i] / 32768;

  const src = playCtx.createBufferSource();
  src.buffer = buf;
  src.connect(playCtx.destination);

  // Never schedule in the past: if the network stalled past the end of what
  // was queued, resume from now instead of dropping the chunk.
  const at = Math.max(playCtx.currentTime, nextStart);
  src.start(at);
  nextStart = at + buf.duration;

  sources.push(src);
  src.onended = () => {
    sources = sources.filter((s) => s !== src);
  };
}

function flushPlayback() {
  for (const s of sources) {
    try {
      s.stop();
    } catch {
      // already ended between the Cancel arriving and this call
    }
  }
  sources = [];
  nextStart = 0;
}

// --- UI -------------------------------------------------------------------

function setStatus(text, state) {
  els.status.textContent = text;
  els.dot.className = 'dot ' + (state || '');
}

function log(kind, text) {
  const line = document.createElement('div');
  line.className = 'line ' + kind;
  const tag = document.createElement('span');
  tag.className = 'tag';
  tag.textContent = kind;
  line.append(tag, document.createTextNode(text));
  els.log.prepend(line);
  while (els.log.childElementCount > 100) els.log.lastElementChild.remove();
}

// --- Session --------------------------------------------------------------

async function start() {
  setStatus('connecting', 'busy');

  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true, // the whole reason this client is a browser
      noiseSuppression: true,
      autoGainControl: false, // AGC would fight the VAD's learned levels
    },
  });

  // 16 kHz context: the browser resamples the mic natively, so the capture
  // worklet can emit server-ready frames with no DSP of our own.
  captureCtx = new AudioContext({ sampleRate: CAPTURE_RATE });
  if (captureCtx.sampleRate !== CAPTURE_RATE) {
    log('warn', `browser gave a ${captureCtx.sampleRate} Hz capture context, not ${CAPTURE_RATE}`);
  }
  await captureCtx.audioWorklet.addModule('capture-worklet.js');

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    seq = 0;
    send({
      streamStart: {
        sampleRateHz: CAPTURE_RATE,
        channels: 1,
        bitsPerSample: 16,
        frameDurationMs: FRAME_MS,
      },
    });

    const source = captureCtx.createMediaStreamSource(micStream);
    const node = new AudioWorkletNode(captureCtx, 'capture');
    node.port.onmessage = (e) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      send({ audioFrame: { seq: seq++, pcm: new Uint8Array(e.data.buffer) } });
    };
    // The worklet has no output; connecting to the destination anyway is what
    // keeps it scheduled in some browsers.
    source.connect(node).connect(captureCtx.destination);

    running = true;
    els.toggle.textContent = 'Stop';
    setStatus('listening', 'live');
  };

  ws.onmessage = (e) => handle(ServerMessage.decode(new Uint8Array(e.data)));
  ws.onerror = () => setStatus('connection error', 'error');
  ws.onclose = () => {
    if (running) stop();
  };
}

function send(msg) {
  ws.send(ClientMessage.encode(ClientMessage.create(msg)).finish());
}

function handle(msg) {
  switch (msg.msg) {
    case 'streamStarted':
      log('stream', `id ${msg.streamStarted.streamId}`);
      break;

    case 'transcript':
      els.transcript.textContent = msg.transcript.text;
      els.reply.textContent = '';
      log('you', msg.transcript.text);
      break;

    case 'replyDelta':
      els.reply.textContent += msg.replyDelta.text;
      break;

    case 'replyAudio':
      playChunk(msg.replyAudio.pcm, msg.replyAudio.sampleRateHz);
      setStatus('speaking — talk over it to interrupt', 'speaking');
      break;

    case 'cancel':
      // Drop everything already buffered. Waiting for replyEnd instead would
      // leave the assistant audibly talking after the user cut in.
      flushPlayback();
      log('barge', 'you interrupted — playback flushed');
      break;

    case 'replyEnd':
      if (msg.replyEnd.reason === 'ERROR') log('error', 'reply failed server-side');
      setStatus('listening', 'live');
      break;

    case 'error':
      log('error', `${msg.error.code}: ${msg.error.message}`);
      setStatus(msg.error.code, 'error');
      break;
  }
}

function stop() {
  running = false;
  flushPlayback();
  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      send({ streamStop: {} });
    } catch {
      // socket died first; nothing to tell it
    }
    ws.close();
  }
  ws = null;
  if (captureCtx) captureCtx.close();
  captureCtx = null;
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  micStream = null;
  els.toggle.textContent = 'Start';
  setStatus('idle');
}

els.toggle.onclick = async () => {
  if (running) {
    stop();
    return;
  }
  try {
    await start();
  } catch (err) {
    setStatus('failed to start', 'error');
    log('error', String(err));
    stop();
  }
};

protobuf.load('astra.proto').then((root) => {
  ClientMessage = root.lookupType('astra.v1.ClientMessage');
  ServerMessage = root.lookupType('astra.v1.ServerMessage');
  els.toggle.disabled = false;
  setStatus('idle');
});
