import asyncio
import websockets
import argparse
import logging
import faster_whisper
from faster_whisper import WhisperModel
import os
import json  # Thêm import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=10197)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--model_size", type=str, default="small")
args = parser.parse_args()

logger.info("Loading Whisper model...")
asr_model = WhisperModel(args.model_size, device=args.device, compute_type="float16")
logger.info("Model loaded.")

task_queue = asyncio.Queue()

async def ws_serve(websocket, path):
    try:
        async for message in websocket:
            if isinstance(message, str):
                data = json.loads(message)  # Sử dụng json.loads
                await task_queue.put((websocket, data.get('audio', b'')))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

async def worker():
    while True:
        websocket, audio_data = await task_queue.get()
        if websocket.open:
            await process_audio(websocket, audio_data)
        task_queue.task_done()

async def process_audio(websocket, audio_data):
    try:
        with open("temp.wav", "wb") as f:
            f.write(audio_data)
        segments, _ = asr_model.transcribe("temp.wav", language="vi", beam_size=5)
        text = " ".join([seg.text for seg in segments]).strip()
        if websocket.open:
            await websocket.send(text)
    except Exception as e:
        logger.error(f"Transcription error: {e}")
    finally:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")

async def run_server():
    async with websockets.serve(ws_serve, args.host, args.port):
        asyncio.create_task(worker())
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(run_server()) 