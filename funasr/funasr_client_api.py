import ssl
from websocket import ABNF, create_connection
from queue import Queue
import threading
import json
import time
import argparse

class Funasr_websocket_recognizer:
    def __init__(self, host="127.0.0.1", port=10197, is_ssl=False):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--host", type=str, default="127.0.0.1")
        self.parser.add_argument("--port", type=int, default=10197)
        self.parser.add_argument("--chunk_size", type=int, default=160)
        self.args = self.parser.parse_args()

        uri = "wss://" if is_ssl else "ws://"
        uri += f"{self.args.host}:{self.args.port}"
        ssl_opt = {"cert_reqs": ssl.CERT_NONE} if is_ssl else None
        self.websocket = create_connection(uri, sslopt=ssl_opt)
        self.msg_queue = Queue()
        threading.Thread(target=self.thread_rec_msg, daemon=True).start()
        self.websocket.send(json.dumps({"mode": "2pass", "language": "vi", "chunk_size": self.args.chunk_size}))

    def thread_rec_msg(self):
        while True:
            msg = self.websocket.recv()
            if msg:
                self.msg_queue.put(json.loads(msg))

    def feed_chunk(self, chunk, wait_time=0.01):
        try:
            self.websocket.send(chunk, ABNF.OPCODE_BINARY)
            return self.msg_queue.get(timeout=wait_time)
        except:
            return {}

    def close(self, timeout=1):
        self.websocket.send(json.dumps({"is_speaking": False}))
        time.sleep(timeout)
        msg = ""
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
        self.websocket.close()
        return msg

if __name__ == "__main__":
    rcg = Funasr_websocket_recognizer()
    # Test code...