import time
import asyncio
import socket
from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.logger import setup_logger
from src.prompt import system_prompt
from src.asr import run_asr
from src.tts import run_tts
import os
import threading
import subprocess
from asr.ASR_server import run_server as asr_main
from flask_cors import CORS
import configparser

app = Flask(__name__)
CORS(app)
load_dotenv()

# Đọc config từ system.conf
config = configparser.ConfigParser()
config.read('system.conf')
ASR_PORT = int(os.environ.get('ASR_PORT', config.get('DEFAULT', 'ASR_PORT', fallback=10197)))
ASR_HOST = os.environ.get('ASR_HOST', config.get('DEFAULT', 'ASR_HOST', fallback='0.0.0.0'))
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', config.get('DEFAULT', 'LLM_BASE_URL', fallback='http://localhost:11434'))
LLM_MODEL = os.environ.get('LLM_MODEL', config.get('DEFAULT', 'LLM_MODEL', fallback='gemma3:4b'))
PINECONE_INDEX = os.environ.get('PINECONE_INDEX', config.get('DEFAULT', 'PINECONE_INDEX', fallback='digital-hunman'))

logger = setup_logger("TourGuideBot")

# Kiểm tra và giải phóng port với retry
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def free_port(port, max_retries=3):
    for attempt in range(max_retries):
        if not is_port_in_use(port):
            return True
        logger.warning(f"Port {port} in use. Attempt {attempt + 1}/{max_retries} to free...")
        try:
            result = subprocess.check_output(['netstat', '-aon', '|', 'findstr', f':{port}'], shell=True, text=True)
            lines = result.strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) > 4 and f':{port}' in parts[2]:
                    pid = parts[-1]
                    if pid and pid != '0':
                        subprocess.run(['taskkill', '/F', '/PID', pid], shell=True, check=True)
                        logger.info(f"Freed port {port} by killing PID {pid}.")
                        time.sleep(2)
                        if not is_port_in_use(port):
                            return True
            logger.info(f"Failed to free port {port} automatically after {max_retries} attempts.")
            return False
        except Exception as e:
            logger.info(f"Error freeing port {port}: {e}. Please free manually (netstat -aon | findstr :{port} then taskkill /PID <PID> /F).")
            return False
    return False

# Khởi động ASR server
def start_asr_server():
    if not free_port(ASR_PORT):
        logger.error(f"Cannot start ASR on port {ASR_PORT}. Skipping.")
        return None
    logger.info(f"Starting ASR server on {ASR_HOST}:{ASR_PORT}...")
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=run_event_loop, args=(loop,), daemon=True)
    thread.start()
    time.sleep(2)  # Chờ ASR khởi động
    return thread

def run_event_loop(loop):
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(asr_main())
    except asyncio.CancelledError:
        logger.info("ASR server cancelled.")
    except Exception as e:
        logger.error(f"ASR server error: {e}")
    finally:
        loop.close()

asr_thread = start_asr_server()

# Khởi tạo RAG
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY not found.")
    exit(1)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

try:
    embeddings = download_hugging_face_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX, embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chat_model = ChatOllama(model=LLM_MODEL, base_url=LLM_BASE_URL)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
except Exception as e:
    logger.error(f"RAG setup failed: {str(e)}")
    exit(1)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")
    if not msg:
        return "No message provided", 400
    try:
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        return answer
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/asr', methods=['POST'])
def asr_endpoint():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    audio_file = request.files['audio']
    audio_data = audio_file.read()
    text = run_asr(audio_data)
    return jsonify({'text': text}) if text else jsonify({'error': 'ASR failed'}), 500

@app.route('/tts', methods=['POST'])
def tts_endpoint():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text'}), 400
    audio_path = run_tts(text)
    return jsonify({'audio_url': audio_path.replace('./', '/static/')}) if audio_path else jsonify({'error': 'TTS failed'}), 500

@app.route('/v1/chat/completions', methods=['POST'])
def openai_chat():
    data = request.json
    messages = data.get('messages', [])
    user_input = messages[-1]['content'] if messages else ''
    response = rag_chain.invoke({"input": user_input})
    answer = response["answer"]
    return jsonify({
        'id': 'chatcmpl-123',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': LLM_MODEL,
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    })

if __name__ == '__main__':
    logger.info("Starting TourGuideBot...")
    try:
        app.run(host="0.0.0.0", port=8080, debug=True)
    except KeyboardInterrupt:
        logger.info("Shutting down TourGuideBot...")
        if asr_thread:
            asr_thread.join(timeout=2)