import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # reads GOOGLE_API_KEY from your .env

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"

def load_documents(docs_dir):
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from '{docs_dir}/'")
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def embed_and_store(chunks):
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print("Embedding chunks and writing to ChromaDB (this may take a minute)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Stored {len(chunks)} chunks in '{CHROMA_DIR}/'")
    return vector_store

if __name__ == "__main__":
    docs = load_documents(DOCS_DIR)
    chunks = chunk_documents(docs)
    embed_and_store(chunks)
    print("Done. Vector store is ready.")