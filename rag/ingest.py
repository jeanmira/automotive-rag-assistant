"""
Ingestion pipeline: PDF -> chunks -> embeddings -> ChromaDB.
This is the "write" side of the RAG system.
"""

import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from rag.config import (
    DATA_DIR,
    EMBEDDINGS_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)


def _detect_category(pdf_path, data_dir):
    """Derive category from subdirectory name (e.g. 'autosar', 'diagnostics')."""
    rel = os.path.relpath(pdf_path, data_dir)
    parts = rel.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return "general"


def load_pdfs(data_dir=None):
    """Walk data/ and load every PDF into LangChain documents with metadata."""
    data_dir = data_dir or DATA_DIR
    pdf_paths = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)

    if not pdf_paths:
        print(f"no PDF files found in {os.path.abspath(data_dir)}")
        return []

    documents = []
    for path in sorted(pdf_paths):
        filename = os.path.basename(path)
        category = _detect_category(path, data_dir)
        print(f"  loading [{category}] {filename}...", end=" ", flush=True)
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            # tag each page so we can trace answers back to source
            for page in pages:
                page.metadata["source_filename"] = filename
                page.metadata["category"] = category
            documents.extend(pages)
            print(f"{len(pages)} pages")
        except Exception as e:
            print(f"failed: {e}")

    return documents


def split_documents(documents):
    """Split pages into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks, persist_dir=None):
    """Embed chunks and store them in ChromaDB."""
    persist_dir = persist_dir or EMBEDDINGS_DIR
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )


def get_vector_store(persist_dir=None):
    """Load existing vector store from disk (read-only access)."""
    persist_dir = persist_dir or EMBEDDINGS_DIR

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def ingest(data_dir=None, persist_dir=None):
    """Full pipeline: load PDFs -> split into chunks -> embed -> store."""
    print("indexing documents...\n")

    documents = load_pdfs(data_dir)
    if not documents:
        return None

    print(f"\n  total pages loaded: {len(documents)}")

    chunks = split_documents(documents)
    print(f"  chunks created: {len(chunks)} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print("  creating embeddings and storing in chromadb...")
    vector_store = create_vector_store(chunks, persist_dir)

    count = vector_store._collection.count()
    print(f"  indexed {count} chunks in collection '{COLLECTION_NAME}'")
    print("\ndone")

    return vector_store


if __name__ == "__main__":
    ingest()
