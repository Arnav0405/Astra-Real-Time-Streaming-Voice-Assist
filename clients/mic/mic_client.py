#!/usr/bin/env python3
"""Stream the local microphone to the astra WebSocket server for manual testing.

Captures 16 kHz mono s16le audio in 20 ms frames (the server's strict format),
protobuf-frames each as an AudioFrame, and streams them. The server does the
VAD / wake-word / endpointing; run it with -verbose to see the pipeline trace.

Stop with 'q' (or Ctrl+Q / Ctrl+C): sends StreamStop and closes cleanly.

Run inside the ml venv (has sounddevice + websockets):
    services/ml/.venv/bin/python clients/mic/mic_client.py
"""
import argparse
import asyncio
import signal
import sys
import termios
import tty
from pathlib import Path

import sounddevice as sd
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astra.v1 import astra_pb2 as pb  # noqa: E402

SR = 16000
FRAME_SAMPLES = 320  # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2  # s16le


async def run(url: str, device):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    stop = asyncio.Event()

    def on_stop(*_):
        loop.call_soon_threadsafe(stop.set)

    signal.signal(signal.SIGINT, on_stop)

    # Mic callback runs in the PortAudio thread — hand bytes to the loop.
    def audio_cb(indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        try:
            loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))
        except asyncio.QueueFull:
            pass  # drop under backpressure; a demo, not a recorder

    stream = sd.RawInputStream(
        samplerate=SR, channels=1, dtype="int16",
        blocksize=FRAME_SAMPLES, device=device, callback=audio_cb,
    )

    print(f"connecting to {url} ...")
    async with websockets.connect(url, max_size=None) as ws:
        start = pb.ClientMessage(stream_start=pb.StreamStart(
            sample_rate_hz=SR, channels=1, bits_per_sample=16, frame_duration_ms=20))
        await ws.send(start.SerializeToString())

        reply = pb.ServerMessage()
        reply.ParseFromString(await ws.recv())
        if reply.HasField("error"):
            print(f"server refused: {reply.error.code}: {reply.error.message}")
            return
        print(f"stream started: {reply.stream_started.stream_id}")
        print("🎙  speak now — stop with 'q' / Ctrl+C\n")

        start_keyboard(loop, stop)
        stream.start()

        async def receiver():
            try:
                while not stop.is_set():
                    msg = pb.ServerMessage()
                    msg.ParseFromString(await ws.recv())
                    if msg.HasField("error"):
                        print(f"⚠ server error: {msg.error.code}: {msg.error.message}")
                        stop.set()
                        return
            except websockets.ConnectionClosed:
                stop.set()

        recv_task = asyncio.create_task(receiver())
        seq = 0
        stop_wait = asyncio.create_task(stop.wait())
        try:
            while not stop.is_set():
                get_task = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {get_task, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
                if get_task in done:
                    pcm = get_task.result()
                    if len(pcm) != FRAME_BYTES:
                        continue
                    frame = pb.ClientMessage(audio_frame=pb.AudioFrame(seq=seq, pcm=pcm))
                    await ws.send(frame.SerializeToString())
                    seq += 1
                else:
                    get_task.cancel()
        finally:
            stream.stop()
            stream.close()
            try:
                await ws.send(pb.ClientMessage(stream_stop=pb.StreamStop()).SerializeToString())
            except websockets.ConnectionClosed:
                pass
            recv_task.cancel()
            print(f"\nstopped — sent {seq} frames.")


def start_keyboard(loop, stop):
    """Set `stop` on q / Ctrl+Q / Ctrl+C typed at the terminal. No-op if stdin
    is not a tty (e.g. piped)."""
    if not sys.stdin.isatty():
        return

    def reader():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop.is_set():
                ch = sys.stdin.read(1)
                if ch in ("q", "\x11", "\x03"):  # q, Ctrl+Q, Ctrl+C
                    loop.call_soon_threadsafe(stop.set)
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    import threading
    threading.Thread(target=reader, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8080", help="astra WebSocket URL")
    ap.add_argument("--device", default=None, help="input device (name or index); default = system mic")
    args = ap.parse_args()
    device = args.device
    if device is not None and device.isdigit():
        device = int(device)
    try:
        asyncio.run(run(args.url, device))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
