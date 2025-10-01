import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List

def load_pdf_file(data):
    try:
        if not os.path.exists(data):
            raise FileNotFoundError(f"Directory not found: {data}")
        loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
        return loader.load()
    except Exception as e:
        raise RuntimeError(f"PDF loading error: {str(e)}")

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    return [Document(page_content=doc.page_content, metadata={"source": doc.metadata.get("source", "")}) for doc in docs]

def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(extracted_data)

def download_hugging_face_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        raise RuntimeError(f"Embeddings download error: {str(e)}")