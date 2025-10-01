from threading import Thread
import websocket
import json
import time
import ssl
from src.logger import setup_logger

logger = setup_logger("FunASR")

class FunASR:
    def __init__(self):
        self.__URL = "ws://127.0.0.1:10197"
        self.__ws = None
        self.__connected = False
        self.__frames = []
        self.done = False
        self.final_results = ""

    def on_message(self, ws, message):
        self.done = True
        self.final_results = message
        logger.info(f"FunASR result: {message}")

    def on_close(self, ws, code, msg):
        self.__connected = False

    def on_error(self, ws, error):
        logger.error(f"FunASR error: {error}")
        self.__connected = False

    def on_open(self, ws):
        self.__connected = True
        def run():
            while self.__connected:
                if self.__frames:
                    frame = self.__frames.pop(0)
                    if isinstance(frame, dict):
                        frame['language'] = 'vi'
                        ws.send(json.dumps(frame))
                    elif isinstance(frame, bytes):
                        ws.send(frame, websocket.ABNF.OPCODE_BINARY)
                time.sleep(0.04)
        Thread(target=run, daemon=True).start()

    def __connect(self):
        self.final_results = ""
        self.done = False
        self.__frames.clear()
        self.__ws = websocket.WebSocketApp(self.__URL, on_message=self.on_message, on_close=self.on_close, on_error=self.on_error)
        self.__ws.on_open = self.on_open
        self.__ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def send(self, buf):
        self.__frames.append(buf)

    def start(self):
        Thread(target=self.__connect).start()
        self.send({"vad_need": False, "state": "StartTranscription"})

    def end(self):
        if self.__connected:
            for frame in self.__frames:
                if isinstance(frame, dict):
                    self.__ws.send(json.dumps(frame))
                elif isinstance(frame, bytes):
                    self.__ws.send(frame, websocket.ABNF.OPCODE_BINARY)
            self.__frames.clear()
            self.__ws.send(json.dumps({"vad_need": False, "state": "StopTranscription"}))
        self.__connected = False

if __name__ == "__main__":
    funasr = FunASR()
    funasr.start()
    # Test code...
    funasr.end()