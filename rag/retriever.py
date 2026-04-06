"""
Retrieval + generation: the "read" side of the RAG system.
Takes a question, finds relevant chunks, and streams an answer from the LLM.
"""

import time
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from rag.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    RETRIEVAL_TOP_K,
    SYSTEM_PROMPT,
    SOURCE_EXCERPT_LENGTH,
)
from rag.ingest import get_vector_store


# constrain the LLM to only use the provided context
PROMPT_TEMPLATE = """Use the following context to answer the question. If you cannot
find the answer in the context, say "I could not find this information in the indexed
documents."

Context:
{context}

Question: {question}

Answer:"""


def _get_retriever(vector_store=None):
    """Wrap the vector store as a LangChain retriever (top-k similarity)."""
    if vector_store is None:
        vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_TOP_K},
    )


def _get_llm():
    """Create an Ollama LLM instance with a 60s timeout to avoid hanging."""
    return OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        timeout=60,
    )


def _build_prompt(context_docs, question):
    """Combine retrieved chunks into a single prompt string."""
    context = "\n\n".join(doc.page_content for doc in context_docs)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def build_chain(vector_store=None):
    """Build a RetrievalQA chain (used by the non-streaming query path)."""
    if vector_store is None:
        vector_store = get_vector_store()

    retriever = _get_retriever(vector_store)
    llm = _get_llm()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def format_sources(source_documents):
    """Extract unique source citations (filename + page + excerpt) from chunks."""
    seen = set()
    sources = []

    for doc in source_documents:
        filename = doc.metadata.get("source_filename", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        key = f"{filename}:p{page}"
        if key in seen:
            continue
        seen.add(key)

        # clean up whitespace in the excerpt
        excerpt = doc.page_content[:SOURCE_EXCERPT_LENGTH].strip()
        excerpt = " ".join(excerpt.split())

        sources.append({
            "filename": filename,
            "page": page,
            "excerpt": excerpt,
        })

    return sources


def retrieve_docs(question, vector_store=None):
    """Find the most relevant chunks for a question (no LLM call)."""
    retriever = _get_retriever(vector_store)
    return retriever.invoke(question)


def stream_answer(question, context_docs):
    """Stream the LLM response token by token. Used by the Streamlit UI."""
    llm = _get_llm()
    prompt_text = _build_prompt(context_docs, question)

    for chunk in llm.stream(prompt_text):
        yield chunk


def query(question, vector_store=None):
    """Non-streaming query -- returns the full answer at once."""
    chain = build_chain(vector_store)

    start = time.time()
    result = chain.invoke({"query": question})
    elapsed = time.time() - start

    return {
        "answer": result["result"],
        "sources": format_sources(result["source_documents"]),
        "elapsed": round(elapsed, 1),
    }
