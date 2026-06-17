import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"

st.set_page_config(page_title="RAG Q&A Assistant", page_icon="📚")
st.title("📚 RAG Q&A Assistant")

@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

@st.cache_resource
def get_vector_store():
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    # If the store is empty (e.g. a fresh cloud deploy), build it from docs/
    if len(store.get()["ids"]) == 0 and os.path.isdir(DOCS_DIR):
        pages = PyPDFDirectoryLoader(DOCS_DIR).load()
        if pages:
            chunks = splitter.split_documents(pages)
            store.add_documents(chunks)
    return store

llm = get_llm()
vector_store = get_vector_store()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

with st.sidebar:
    st.header("Add documents")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process documents") and uploaded:
        with st.spinner("Reading, chunking, embedding..."):
            total = 0
            for uf in uploaded:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                pages = PyPDFLoader(tmp_path).load()
                chunks = splitter.split_documents(pages)
                for c in chunks:
                    c.metadata["source"] = uf.name
                vector_store.add_documents(chunks)
                total += len(chunks)
                os.unlink(tmp_path)
        st.success(f"Added {total} chunks from {len(uploaded)} file(s).")

question = st.text_input("Ask a question about your documents:")
if question:
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    if not docs:
        st.warning("No documents yet — upload a PDF in the sidebar.")
    else:
        context = format_docs(docs)
        answer = (prompt | llm | StrOutputParser()).invoke(
            {"context": context, "question": question}
        )
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Sources")
        for i, d in enumerate(docs, 1):
            src = os.path.basename(str(d.metadata.get("source", "?")))
            page = d.metadata.get("page", "?")
            with st.expander(f"Source {i} — {src}, page {page}"):
                st.write(d.page_content)