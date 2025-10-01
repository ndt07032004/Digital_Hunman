# import asyncio
# import websockets
# import argparse
# import json
# import logging
# from funasr import AutoModel
# import os

# # 设置日志级别
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.CRITICAL)

# # 解析命令行参数
# parser = argparse.ArgumentParser()
# parser.add_argument("--host", type=str, default="0.0.0.0", help="host ip, localhost, 0.0.0.0")
# parser.add_argument("--port", type=int, default=10197, help="grpc server port")
# parser.add_argument("--ngpu", type=int, default=1, help="0 for cpu, 1 for gpu")
# args = parser.parse_args()

# # 初始化模型
# print("model loading")
# asr_model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
#                       vad_model="fsmn-vad", vad_model_revision="v2.0.4",
#                       punc_model="ct-punc-c", punc_model_revision="v2.0.4")
# print("model loaded")
# websocket_users = {}
# task_queue = asyncio.Queue()

# async def ws_serve(websocket, path):
#     global websocket_users
#     user_id = id(websocket)
#     websocket_users[user_id] = websocket
#     try:
#         async for message in websocket:
#             if isinstance(message, str):
#                 data = json.loads(message)
#                 if 'url' in data:
#                     await task_queue.put((websocket, data['url']))
#     except websockets.exceptions.ConnectionClosed as e:
#         logger.info(f"Connection closed: {e.reason}")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#     finally:
#         logger.info(f"Cleaning up connection for user {user_id}")
#         if user_id in websocket_users:
#             del websocket_users[user_id]
#         await websocket.close()
#         logger.info("WebSocket closed")

# async def worker():
#     while True:
#         websocket, url = await task_queue.get()
#         if websocket.open:
#             await process_wav_file(websocket, url)
#         else:
#             logger.info("WebSocket connection is already closed when trying to process file")
#         task_queue.task_done()

# async def process_wav_file(websocket, url):
#     # 热词
#     param_dict = {"sentence_timestamp": False}
#     with open("data/hotword.txt", "r", encoding="utf-8") as f:
#         lines = f.readlines()
#         lines = [line.strip() for line in lines]
#     hotword = " ".join(lines)
#     print(f"热词：{hotword}")
#     param_dict["hotword"] = hotword
#     wav_path = url
#     try:
#         res = asr_model.generate(input=wav_path, is_final=True, **param_dict)
#         if res:
#             if 'text' in res[0] and websocket.open:
#                 await websocket.send(res[0]['text'])
#     except Exception as e:
#         print(f"Error during model.generate: {e}")
#     finally:
#         if os.path.exists(wav_path):
#             os.remove(wav_path)

# async def main():
#     start_server = websockets.serve(ws_serve, args.host, args.port, ping_interval=10)
#     await start_server
#     worker_task = asyncio.create_task(worker())
#     await worker_task

# # 使用 asyncio 运行主函数
# asyncio.run(main())


import asyncio
import websockets
import argparse
import json
import logging
import os
from faster_whisper import WhisperModel

# =======================
# Logging
# =======================
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

# =======================
# Command-line arguments
# =======================
parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0", help="host ip, localhost, 0.0.0.0")
parser.add_argument("--port", type=int, default=10197, help="grpc server port")
parser.add_argument("--device", type=str, default="cuda", help="cuda hoặc cpu")
parser.add_argument("--model_size", type=str, default="small", help="Whisper model size: tiny, base, small, medium, large-v2")
parser.add_argument("--compute_type", type=str, default="float16", help="float16 cho GPU, int8 cho CPU")
args = parser.parse_args()

# =======================
# Load Whisper model
# =======================
print("model loading")
asr_model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)
print("model loaded")

# =======================
# WebSocket logic
# =======================
websocket_users = {}
task_queue = asyncio.Queue()

async def ws_serve(websocket, path):
    """Xử lý kết nối WebSocket"""
    global websocket_users
    user_id = id(websocket)
    websocket_users[user_id] = websocket
    try:
        async for message in websocket:
            if isinstance(message, str):
                data = json.loads(message)
                if 'url' in data:
                    await task_queue.put((websocket, data['url']))
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed: {e.reason}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info(f"Cleaning up connection for user {user_id}")
        if user_id in websocket_users:
            del websocket_users[user_id]
        await websocket.close()
        logger.info("WebSocket closed")

async def worker():
    """Worker lấy task từ queue"""
    while True:
        websocket, url = await task_queue.get()
        if websocket.open:
            await process_wav_file(websocket, url)
        else:
            logger.info("WebSocket connection closed before processing")
        task_queue.task_done()

async def process_wav_file(websocket, url):
    """Nhận diện file wav bằng Whisper"""
    wav_path = url
    try:
        segments, info = asr_model.transcribe(wav_path, language="vi", beam_size=5)
        text = " ".join([seg.text for seg in segments]).strip()

        if websocket.open:
            await websocket.send(text)
            logger.info(f"Sent result: {text}")
    except Exception as e:
        print(f"Error during model inference: {e}")
        if websocket.open:
            await websocket.send(f"Error: {e}")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

async def main():
    start_server = websockets.serve(ws_serve, args.host, args.port, ping_interval=10)
    await start_server
    worker_task = asyncio.create_task(worker())
    await worker_task

# =======================
# Run server
# =======================
asyncio.run(main())
