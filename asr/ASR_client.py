import pyaudio
import websockets
import asyncio
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=10197)
parser.add_argument("--chunk_size", type=int, default=160)
args = parser.parse_args()

async def record_and_send():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=args.chunk_size)
    uri = f"ws://{args.host}:{args.port}"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"vad_need": True, "language": "vi"}))
        while True:
            data = stream.read(args.chunk_size)
            await websocket.send(data)
            response = await websocket.recv()
            print(f"ASR: {response}")

if __name__ == "__main__":
    asyncio.run(record_and_send())