from src.logger import setup_logger
import os
from dotenv import load_dotenv
import asyncio
import websockets
from src.config import ASR_CONFIG

logger = setup_logger("ASR")

class ASRIntegrator:
    def __init__(self):
        load_dotenv()
        self.host = os.environ.get('ASR_HOST', ASR_CONFIG['host'])
        self.port = os.environ.get('ASR_PORT', ASR_CONFIG['port'])
        self.uri = f"ws://{self.host}:{self.port}"

    async def recognize_speech(self, audio_data):
        try:
            async with websockets.connect(self.uri) as websocket:
                await websocket.send(audio_data)
                return await websocket.recv()
        except Exception as e:
            logger.error(f"ASR connection error: {str(e)}")
            return None

def run_asr(audio_data):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(ASRIntegrator().recognize_speech(audio_data))