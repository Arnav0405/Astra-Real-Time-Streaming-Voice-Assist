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
  waterfall: document.getElementById('waterfall'),
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

// --- Latency waterfall ----------------------------------------------------
//
// Every boundary in a Turn message was stamped on the server, so the bars end
// at "written to the socket". What happens after that — this page decoding,
// scheduling and playing the audio — runs on a different clock and is not in
// the picture.
//
// The two chains are drawn on separate scales on purpose. A ~200 ms barge-in
// next to a ~2 s turn would be a sliver, so the barge block is normalised to
// its own width and says so, rather than being silently stretched.

const MAX_TURNS = 5;
const turnBlocks = new Map(); // utterance_id -> element

function bar(name, startMs, durMs, total, muted) {
  const row = document.createElement('div');
  row.className = 'span' + (muted ? ' muted' : '');

  const label = document.createElement('span');
  label.className = 'name';
  label.textContent = name;

  const track = document.createElement('div');
  track.className = 'track';
  const fill = document.createElement('div');
  fill.className = 'fill';
  fill.style.marginLeft = (100 * startMs / total) + '%';
  fill.style.width = (100 * durMs / total) + '%';
  track.append(fill);

  const ms = document.createElement('span');
  ms.className = 'ms';
  ms.textContent = durMs + ' ms';

  row.append(label, track, ms);
  return row;
}

function chainWidth(spans) {
  // Bars are positioned against the chain's own span, never a fixed scale.
  return Math.max(1, ...spans.map((s) => (s.startMs || 0) + (s.durMs || 0)));
}

function renderTurn(t) {
  const spans = t.spans || [];
  if (!spans.length) return;
  const uid = String(t.utteranceId);

  if (t.chain === 'barge') {
    const block = turnBlocks.get(uid);
    if (!block || block.querySelector('.barge-block')) return;
    const total = chainWidth(spans);
    const wrap = document.createElement('div');
    wrap.className = 'barge-block';
    const head = document.createElement('div');
    head.className = 'barge-head';
    head.innerHTML = '✋ barged <span>— own scale, ' + total + ' ms full width</span>';
    wrap.append(head, ...spans.map((s) => bar(s.name, s.startMs || 0, s.durMs || 0, total, false)));
    block.append(wrap);
    return;
  }

  const total = chainWidth(spans);
  const block = document.createElement('div');
  block.className = 'turn';

  // A failed turn has no tts_ttfb span — the chain stops where it died — so
  // "to first audio" would be a lie. The speech duration is shown either way:
  // how long the user talked is context, not latency.
  const speech = spans.find((s) => s.name === 'user_speech');
  const failed = !spans.some((s) => s.name === 'tts_ttfb');
  const head = document.createElement('div');
  head.className = 'turn-head';
  head.innerHTML = '<span>turn ' + uid + '</span><span>' +
    (speech ? '<b>' + speech.durMs + ' ms</b> spoken · ' : '') +
    (failed
      ? '<em class="failed">failed — no reply</em>'
      : '<b>' + (t.headlineMs || 0) + ' ms</b> to first audio') +
    '</span>';
  block.append(head, ...spans.map((s) =>
    bar(s.name, s.startMs || 0, s.durMs || 0, total, s.name === 'user_speech')));

  els.waterfall.prepend(block);
  turnBlocks.set(uid, block);
  while (els.waterfall.childElementCount > MAX_TURNS) {
    const oldest = els.waterfall.lastElementChild;
    for (const [k, v] of turnBlocks) if (v === oldest) turnBlocks.delete(k);
    oldest.remove();
  }
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
      // The server coalesces: every message carries the full text so far.
      els.transcript.textContent = msg.transcript.text;
      els.transcript.classList.toggle('partial', !msg.transcript.isFinal);
      els.reply.textContent = '';
      if (msg.transcript.isFinal) log('you', msg.transcript.text);
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

    case 'turn':
      renderTurn(msg.turn);
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
