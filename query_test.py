from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"

# SAME embedding model used to build the store
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Load the existing store (note: we don't rebuild, we connect to it)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

query = "What is a pullback?"   # change this to something your book covers
results = vector_store.similarity_search(query, k=4)

print(f"Query: {query}\n")
for i, doc in enumerate(results, 1):
    page = doc.metadata.get("page", "?")
    print(f"--- Result {i}  (page {page}) ---")
    print(doc.page_content[:300])
    print()