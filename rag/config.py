"""Central config for the RAG pipeline. All tunables live here."""

import os
from dotenv import load_dotenv

load_dotenv()

# Ollama connection -- override via .env if needed
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "embeddings")
COLLECTION_NAME = "automotive_docs"

# chunking -- 1000 chars with 200 overlap keeps enough context per chunk
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# how many chunks to feed the LLM as context
RETRIEVAL_TOP_K = 5

# how much of each chunk to show in the UI source citations
SOURCE_EXCERPT_LENGTH = 200

SYSTEM_PROMPT = (
    "You are a technical assistant specialized in automotive software engineering. "
    "Answer questions using ONLY the provided context from indexed documents. "
    "If the context does not contain enough information to answer, respond with: "
    '"I could not find this information in the indexed documents." '
    "Always cite the source document name and page number when possible."
)
