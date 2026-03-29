import os
import time
import streamlit as st
import requests

from rag.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    DATA_DIR,
    EMBEDDINGS_DIR,
    COLLECTION_NAME,
)
from rag.ingest import ingest, get_vector_store
from rag.retriever import retrieve_docs, stream_answer, format_sources


# -- wheel animation SVGs --

WHEEL_PROCESSING = """
<div id="wheel-wrapper" style="display:flex; align-items:center; gap:14px; padding:16px 0;">
  <svg id="wheel" width="48" height="48" viewBox="0 0 48 48"
       style="animation: spin 0.7s linear infinite; flex-shrink:0;">
    <circle cx="24" cy="24" r="22" fill="none" stroke="#444" stroke-width="3"/>
    <circle cx="24" cy="24" r="5" fill="#333" stroke="#555" stroke-width="1.5"/>
    <line x1="24" y1="19" x2="24" y2="4"  stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <line x1="24" y1="29" x2="24" y2="44" stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <line x1="19" y1="24" x2="4"  y2="24" stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <line x1="29" y1="24" x2="44" y2="24" stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <line x1="20.1" y1="20.1" x2="8.1"  y2="8.1"  stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <line x1="27.9" y1="27.9" x2="39.9" y2="39.9" stroke="#444" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <span style="font-family:monospace; font-size:12px; color:#555;">processing...</span>
</div>
<style>
  @keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }
</style>
"""

WHEEL_DONE = """
<div id="wheel-wrapper" style="display:flex; align-items:center; gap:14px; padding:16px 0; overflow:hidden;">
  <svg id="wheel" width="48" height="48" viewBox="0 0 48 48"
       style="animation: spin 0.4s linear infinite, driveoff 0.9s ease-in forwards; flex-shrink:0;">
    <circle cx="24" cy="24" r="22" fill="none" stroke="#444" stroke-width="3"/>
    <circle cx="24" cy="24" r="5" fill="#333" stroke="#555" stroke-width="1.5"/>
    <line x1="24" y1="19" x2="24" y2="4"  stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <line x1="24" y1="29" x2="24" y2="44" stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <line x1="19" y1="24" x2="4"  y2="24" stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <line x1="29" y1="24" x2="44" y2="24" stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <line x1="20.1" y1="20.1" x2="8.1"  y2="8.1"  stroke="#444" stroke-width="2" stroke-linecap="round"/>
    <line x1="27.9" y1="27.9" x2="39.9" y2="39.9" stroke="#444" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <span style="font-family:monospace; font-size:12px; color:#333; animation: fadeout 0.9s ease-in forwards;">done</span>
</div>
<style>
  @keyframes spin    { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
  @keyframes driveoff { 0%{transform:translateX(0) rotate(0deg)} 100%{transform:translateX(400px) rotate(720deg); opacity:0} }
  @keyframes fadeout { 0%{opacity:1} 60%{opacity:1} 100%{opacity:0} }
</style>
"""

PIPELINE_STEPS = ["query", "embed", "retrieve", "generate", "answer"]


def _render_pipeline(pipeline_placeholder, active_index):
    parts = []
    for j, s in enumerate(PIPELINE_STEPS):
        if j == active_index:
            parts.append(f'<span style="color:#1D9E75;font-weight:500">{s}</span>')
        else:
            parts.append(f'<span style="color:#333">{s}</span>')
    joined = " &#x203A; ".join(parts)
    pipeline_placeholder.markdown(
        f'<p style="font-family:monospace;font-size:11px;letter-spacing:.05em">{joined}</p>',
        unsafe_allow_html=True,
    )


def _render_pipeline_done(pipeline_placeholder):
    parts = [f'<span style="color:#1D9E75">{s}</span>' for s in PIPELINE_STEPS]
    joined = " &#x203A; ".join(parts)
    pipeline_placeholder.markdown(
        f'<p style="font-family:monospace;font-size:11px">{joined}</p>',
        unsafe_allow_html=True,
    )


def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def get_indexed_count():
    try:
        vs = get_vector_store()
        return vs._collection.count()
    except Exception:
        return 0


def get_pdf_list():
    if not os.path.isdir(DATA_DIR):
        return []
    import glob as _glob
    paths = _glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
    return sorted(os.path.relpath(p, DATA_DIR) for p in paths)


def inject_css():
    st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  background-color: #0d0d0d !important;
  color: #e5e5e5 !important;
  font-family: 'Geist Mono', 'JetBrains Mono', 'Courier New', monospace !important;
}

[data-testid="stSidebar"] {
  background-color: #111111 !important;
  border-right: 1px solid #222222 !important;
}

.stTextInput input, .stTextArea textarea {
  background-color: #1a1a1a !important;
  border: 1px solid #333333 !important;
  color: #e5e5e5 !important;
  font-family: inherit !important;
  border-radius: 6px !important;
}

.stButton > button,
.stFormSubmitButton > button {
  background-color: #1a1a1a !important;
  color: #e5e5e5 !important;
  border: 1px solid #333333 !important;
  border-radius: 6px !important;
  font-family: inherit !important;
  font-size: 13px !important;
  padding: 6px 16px !important;
  transition: border-color 0.15s !important;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
  border-color: #666666 !important;
  background-color: #222222 !important;
}

.answer-card {
  background: #111111;
  border: 1px solid #222222;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 12px 0;
  font-size: 14px;
  line-height: 1.7;
  color: #e5e5e5;
}

.source-block {
  background: #0a0a0a;
  border: 1px solid #1f1f1f;
  border-radius: 6px;
  padding: 12px 16px;
  font-size: 12px;
  color: #888888;
  margin-top: 8px;
  font-family: monospace;
  white-space: pre-wrap;
}

.response-meta {
  font-size: 11px;
  color: #444444;
  margin-top: 8px;
  text-align: right;
  font-family: monospace;
}

.offline-banner {
  background: #1a1200;
  border: 1px solid #3a2a00;
  border-radius: 6px;
  padding: 10px 16px;
  color: #aa8800;
  font-size: 12px;
  margin-bottom: 16px;
  font-family: monospace;
}

.sidebar-label {
  font-size: 11px;
  color: #444444;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* mobile responsive */
@media (max-width: 768px) {
  .block-container {
    padding: 1rem 0.75rem !important;
  }

  .answer-card {
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.6;
    word-break: break-word;
  }

  .source-block {
    padding: 10px 12px;
    font-size: 11px;
    word-break: break-word;
  }

  .stTextInput input, .stTextArea textarea {
    font-size: 16px !important; /* prevents iOS zoom on focus */
  }

  .stButton > button,
  .stFormSubmitButton > button {
    width: 100% !important;
    padding: 10px 16px !important;
    font-size: 14px !important;
  }

  .offline-banner {
    font-size: 11px;
    padding: 8px 12px;
  }

  h2 {
    font-size: 18px !important;
  }

  /* stack form columns vertically on mobile */
  [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
  }
  [data-testid="stForm"] [data-testid="stHorizontalBlock"] > div {
    width: 100% !important;
    flex: 1 1 100% !important;
  }
}
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-label">documents</div>', unsafe_allow_html=True)

        pdfs = get_pdf_list()
        if pdfs:
            for pdf in pdfs:
                st.markdown(
                    f'<div style="color:#888; font-size:13px; font-family:monospace; '
                    f'margin-bottom:4px;">&#x203A; {pdf}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#555; font-size:12px;">no PDFs in data/</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("index documents"):
            with st.spinner("indexing..."):
                try:
                    vs = ingest()
                    if vs:
                        st.session_state["vector_store"] = vs
                        st.session_state["chunk_count"] = vs._collection.count()
                        st.success("indexing complete")
                    else:
                        st.error("no documents found to index")
                except Exception as e:
                    st.error(f"indexing failed: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">model</div>', unsafe_allow_html=True)

        ollama_online = check_ollama()
        status_color = "#4a4" if ollama_online else "#a44"
        status_text = "ready" if ollama_online else "offline"
        st.markdown(
            f'<div style="color:#888; font-size:13px; font-family:monospace;">'
            f'{LLM_MODEL}  <span style="color:{status_color};">&#x25CF;</span>  {status_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">indexed</div>', unsafe_allow_html=True)

        chunk_count = st.session_state.get("chunk_count", get_indexed_count())
        st.session_state["chunk_count"] = chunk_count
        st.markdown(
            f'<div style="color:#888; font-size:13px; font-family:monospace;">'
            f'{chunk_count} chunks</div>',
            unsafe_allow_html=True,
        )


def render_answer(result):
    answer = result["answer"]
    sources = result["sources"]
    elapsed = result["elapsed"]

    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

    if sources:
        with st.expander(f"sources ({len(sources)})"):
            for src in sources:
                page_display = src["page"] if src["page"] != "?" else "n/a"
                st.markdown(
                    f'<div class="source-block">'
                    f'&#x203A; {src["filename"]}  p.{page_display}\n'
                    f'"{src["excerpt"]}"'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown(
        f'<div class="response-meta">answered in {elapsed}s</div>',
        unsafe_allow_html=True,
    )


def process_question(question, vs):
    """Run the full RAG pipeline with wheel + pipeline animations and streaming output."""
    start = time.time()

    wheel_placeholder = st.empty()
    pipeline_placeholder = st.empty()
    stream_placeholder = st.empty()

    wheel_placeholder.markdown(WHEEL_PROCESSING, unsafe_allow_html=True)

    # step 0: query
    _render_pipeline(pipeline_placeholder, 0)
    time.sleep(0.4)

    # step 1: embed
    _render_pipeline(pipeline_placeholder, 1)
    time.sleep(0.4)

    # step 2: retrieve
    _render_pipeline(pipeline_placeholder, 2)
    context_docs = retrieve_docs(question, vector_store=vs)
    sources = format_sources(context_docs)

    # step 3: generate (streaming)
    _render_pipeline(pipeline_placeholder, 3)

    collected_tokens = []
    with stream_placeholder.container():
        streamed = st.write_stream(stream_answer(question, context_docs))
    collected_tokens.append(streamed)

    # step 4: answer
    _render_pipeline_done(pipeline_placeholder)
    time.sleep(0.5)
    pipeline_placeholder.empty()

    # wheel drives off
    wheel_placeholder.markdown(WHEEL_DONE, unsafe_allow_html=True)
    time.sleep(1.0)
    wheel_placeholder.empty()

    # clear the streamed output -- it will be re-rendered as a proper answer card on rerun
    stream_placeholder.empty()

    elapsed = round(time.time() - start, 1)
    answer_text = "".join(collected_tokens)

    return {
        "answer": answer_text,
        "sources": sources,
        "elapsed": elapsed,
    }


def main():
    st.set_page_config(
        page_title="automotive-rag-assistant",
        page_icon=None,
        layout="wide",
    )

    inject_css()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    render_sidebar()

    st.markdown(
        '<h2 style="color:#e5e5e5; font-family:monospace; font-weight:400; '
        'border-bottom:1px solid #222; padding-bottom:12px;">'
        'automotive-rag-assistant</h2>',
        unsafe_allow_html=True,
    )

    if not check_ollama():
        st.markdown(
            '<div class="offline-banner">'
            'ollama is not running -- start it with: ollama serve'
            '</div>',
            unsafe_allow_html=True,
        )

    # display chat history
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div style="color:#888; font-size:13px; font-family:monospace; '
                f'margin:16px 0 4px 0;">&gt; {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            render_answer(msg["result"])

    with st.form("question_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "question",
                placeholder="ask about the docs...",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("ask")

    clear_col1, clear_col2 = st.columns([6, 1])
    with clear_col2:
        if st.button("clear"):
            st.session_state["messages"] = []
            st.rerun()

    if submitted and user_input and user_input.strip():
        question = user_input.strip()

        if not check_ollama():
            st.markdown(
                '<div class="offline-banner">'
                'cannot query: ollama is offline'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        chunk_count = st.session_state.get("chunk_count", 0)
        if chunk_count == 0:
            st.markdown(
                '<div class="offline-banner">'
                'no documents indexed -- click "index documents" in the sidebar first'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        st.session_state["messages"].append({"role": "user", "content": question})

        try:
            vs = st.session_state.get("vector_store")
            if vs is None:
                vs = get_vector_store()
                st.session_state["vector_store"] = vs

            result = process_question(question, vs)
            st.session_state["messages"].append({"role": "assistant", "result": result})
            st.rerun()

        except Exception as e:
            error_msg = str(e)
            st.markdown(
                f'<div class="offline-banner">error: {error_msg}</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
