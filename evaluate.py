import shutil
import uuid
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
DOCS_DIR = "docs"

# Your FIXED test set. Pick questions you know the answers to, and for each,
# 1-2 essential words that ANY correct answer must contain. Add ~8-10 total.
TEST_SET = [
    {"q": "What is tokenization?",                          "keywords": ["token"]},
    {"q": "What is an embedding?",                          "keywords": ["vector"]},
    {"q": "What is self-attention used for?",               "keywords": ["token"]},
    {"q": "What is the transformer architecture?",          "keywords": ["attention"]},
    {"q": "What happens during pretraining?",               "keywords": ["predict"]},
    {"q": "What is RLHF?",                                   "keywords": ["feedback"]},
    {"q": "What does the temperature setting control?",     "keywords": ["random"]},
    {"q": "What is a context window measured in?",          "keywords": ["token"]},
    {"q": "Why do language models hallucinate?",            "keywords": ["fact"]},
    {"q": "What is the knowledge cutoff?",                  "keywords": ["training"]},
]

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_store(chunk_size, chunk_overlap):
    persist_dir = f"chroma_eval_{uuid.uuid4().hex[:8]}"  # unique each time
    pages = PyPDFDirectoryLoader(DOCS_DIR).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    store = Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        collection_name="eval", persist_directory=persist_dir,
    )
    return store, len(chunks)

def safe_invoke(chain, payload, max_retries=5):
    for attempt in range(max_retries):
        try:
            return chain.invoke(payload)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("   ...rate limited, waiting 30s")
                time.sleep(30)
            else:
                raise
    raise RuntimeError("Still rate limited after retries")

def run_config(chunk_size=1000, chunk_overlap=150, k=4):
    store, n_chunks = build_store(chunk_size, chunk_overlap)
    retriever = store.as_retriever(search_kwargs={"k": k})
    hits = 0
    for item in TEST_SET:
        docs = retriever.invoke(item["q"])
        context = " ".join(d.page_content.lower() for d in docs)
        ok = all(kw.lower() in context for kw in item["keywords"])
        hits += ok
        print(f"[{'HIT ' if ok else 'MISS'}] {item['q']}")
    score = hits / len(TEST_SET)
    print(f"\nconfig: chunk_size={chunk_size}, overlap={chunk_overlap}, k={k}, chunks={n_chunks}")
    print(f"RETRIEVAL SCORE: {hits}/{len(TEST_SET)} = {score:.0%}\n")
    return score

if __name__ == "__main__":
    import glob
    for d in glob.glob("chroma_eval_*"):
        shutil.rmtree(d, ignore_errors=True)

    configs = [
        {"chunk_size": 1500, "chunk_overlap": 150, "k": 1},  # weak default
        {"chunk_size": 1000, "chunk_overlap": 150, "k": 2},
        {"chunk_size": 1000, "chunk_overlap": 150, "k": 4},  # original baseline
        {"chunk_size": 500,  "chunk_overlap": 100, "k": 6},  # tuned
    ]
    results = []
    for cfg in configs:
        results.append((cfg, run_config(**cfg)))

    print("==== SUMMARY ====")
    for cfg, score in results:
        print(f"chunk={cfg['chunk_size']:>4}  overlap={cfg['chunk_overlap']:>3}  k={cfg['k']}  ->  {score:.0%}")