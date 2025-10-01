from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.logger import setup_logger
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index
index_name = "digital-hunman"  # dùng index có sẵn hoặc tạo mới nếu chưa có
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# LangChain RAG setup
chatModel = ChatOllama(model="gemma3:4b", base_url="http://localhost:11434")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])



logger = setup_logger()
logger.info("Rag bắt đầu chạy")
logger.debug("gỡ lỗi")
logger.error("lỗi")
logger.critical("lỗi nghiêm trọng")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
