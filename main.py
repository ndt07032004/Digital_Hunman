import time
import asyncio
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
from asr.ASR_server import run_server as asr_main
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
load_dotenv()

logger = setup_logger("TourGuideBot")

# Khởi động ASR server với asyncio
def start_asr_server():
    logger.info("Starting ASR server...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    threading.Thread(target=loop.run_until_complete, args=(asr_main(),), daemon=True).start()

start_asr_server()

# Khởi tạo RAG
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY not found.")
    exit(1)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

try:
    embeddings = download_hugging_face_embeddings()
    index_name = "digital-hunman"
    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chat_model = ChatOllama(model="gemma3:4b", base_url="http://localhost:11434")
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
        'model': 'gemma3:4b',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    })

if __name__ == '__main__':
    logger.info("Starting TourGuideBot...")
    app.run(host="0.0.0.0", port=8080, debug=True)