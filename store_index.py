from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

try:
    extracted_data = load_pdf_file('data/')
    filtered_data = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(filtered_data)
    embeddings = download_hugging_face_embeddings()
except Exception as e:
    raise RuntimeError(f"Data processing failed: {str(e)}")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "digital-hunman"  # Sửa tên index thành digital-hunman

if not pc.has_index(index_name):
    pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

try:
    PineconeVectorStore.from_documents(text_chunks, index_name=index_name, embedding=embeddings)
    print("Index created successfully.")
except Exception as e:
    raise RuntimeError(f"Vector store creation failed: {str(e)}")