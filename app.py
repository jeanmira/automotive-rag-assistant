"""
Streamlit UI for the automotive RAG assistant.
Dark theme with glassmorphism, animated pipeline steps, and streaming answers.
"""

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


# -- thinking animation shown while the LLM is generating --

WHEEL_PROCESSING = """
<div style="display:flex; align-items:center; gap:12px; padding:16px 0;">
  <div style="display:flex; gap:6px; align-items:center; background:rgba(59,130,246,0.08); backdrop-filter:blur(8px); border:1px solid rgba(59,130,246,0.15); border-radius:20px; padding:10px 20px;">
    <div style="width:6px; height:6px; background:#3b82f6; border-radius:50%; animation:bounce 1.2s ease-in-out infinite;"></div>
    <div style="width:6px; height:6px; background:#3b82f6; border-radius:50%; animation:bounce 1.2s ease-in-out 0.2s infinite;"></div>
    <div style="width:6px; height:6px; background:#3b82f6; border-radius:50%; animation:bounce 1.2s ease-in-out 0.4s infinite;"></div>
    <span style="font-family:monospace; font-size:11px; color:#94a3b8; margin-left:10px;">thinking</span>
  </div>
</div>
"""

WHEEL_DONE = """
<div style="display:flex; align-items:center; gap:12px; padding:16px 0; animation:fadeout 0.8s ease-in forwards;">
  <div style="display:flex; gap:4px; align-items:center; background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.2); border-radius:20px; padding:10px 20px;">
    <span style="color:#06b6d4; font-size:13px;">&#x2713;</span>
    <span style="font-family:monospace; font-size:11px; color:#06b6d4; margin-left:4px;">done</span>
  </div>
</div>
"""

# visual steps shown during RAG processing
PIPELINE_STEPS = ["query", "embed", "retrieve", "generate", "answer"]

# category display config -- using unicode chars (not HTML entities) for expander titles
CATEGORY_STYLES = {
    "autosar": {"icon": "\u2699", "color": "#3b82f6", "bg": "rgba(59,130,246,0.1)"},
    "cybersecurity": {"icon": "\U0001f6e1", "color": "#f59e0b", "bg": "rgba(245,158,11,0.1)"},
    "diagnostics": {"icon": "\u2692", "color": "#10b981", "bg": "rgba(16,185,129,0.1)"},
    "functional_safety": {"icon": "\u26a0", "color": "#ef4444", "bg": "rgba(239,68,68,0.1)"},
}

# suggested questions shown when chat is empty
SUGGESTED_QUESTIONS = [
    "What are the ASIL levels in ISO 26262?",
    "Explain the AUTOSAR layered architecture",
    "What are the main CAN bus vulnerabilities?",
    "How does OBD-II diagnostics work?",
]


def _render_pipeline(pipeline_placeholder, active_index):
    parts = []
    for j, s in enumerate(PIPELINE_STEPS):
        if j == active_index:
            parts.append(
                f'<span style="background:linear-gradient(135deg,rgba(59,130,246,0.2),rgba(6,182,212,0.2)); color:#06b6d4; padding:3px 12px; border-radius:12px; font-weight:500; box-shadow:0 0 12px rgba(6,182,212,0.3); animation:pulse 1.5s ease-in-out infinite;">{s}</span>'
            )
        elif j < active_index:
            parts.append(f'<span style="color:#06b6d4; padding:3px 12px;">{s}</span>')
        else:
            parts.append(f'<span style="color:#475569; padding:3px 12px;">{s}</span>')
    joined = " &#x203A; ".join(parts)
    pipeline_placeholder.markdown(
        f'<p style="font-family:monospace;font-size:11px;letter-spacing:.05em">{joined}</p>',
        unsafe_allow_html=True,
    )


def _render_pipeline_done(pipeline_placeholder):
    parts = [f'<span style="color:#06b6d4; padding:3px 12px;">{s}</span>' for s in PIPELINE_STEPS]
    joined = " &#x203A; ".join(parts)
    pipeline_placeholder.markdown(
        f'<p style="font-family:monospace;font-size:11px">{joined}</p>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=10, show_spinner=False)
def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@st.cache_data(ttl=30, show_spinner=False)
def get_indexed_count():
    """Count indexed chunks using ChromaDB client directly (no embedding init)."""
    try:
        import chromadb
        if not os.path.isdir(EMBEDDINGS_DIR):
            return 0
        client = chromadb.PersistentClient(path=EMBEDDINGS_DIR)
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


@st.cache_resource(show_spinner=False)
def _cached_vector_store():
    """Load vector store once and reuse across reruns."""
    return get_vector_store()


def get_pdf_list():
    if not os.path.isdir(DATA_DIR):
        return []
    import glob as _glob
    paths = _glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
    return sorted(os.path.relpath(p, DATA_DIR) for p in paths)


def inject_css():
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

.block-container {
  padding-top: 1rem !important;
}

header[data-testid="stHeader"] {
  display: none !important;
}

div[data-testid="stAppViewContainer"] {
  padding-top: 0 !important;
}

html, body, [data-testid="stAppViewContainer"] {
  background-color: #0f1117 !important;
  color: #f1f5f9 !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f1117 0%, #131620 100%) !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
  background: rgba(26,29,46,0.5);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  margin-bottom: 6px;
  transition: all 0.2s ease;
}

[data-testid="stSidebar"] [data-testid="stExpander"]:hover {
  border-color: rgba(59,130,246,0.3);
  background: rgba(26,29,46,0.8);
}

.stTextInput input, .stTextArea textarea {
  background-color: #1a1d2e !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  color: #f1f5f9 !important;
  font-family: 'Inter', sans-serif !important;
  border-radius: 10px !important;
  padding: 12px 16px !important;
  font-size: 14px !important;
  transition: all 0.3s ease !important;
}

.stTextInput input::placeholder {
  color: #64748b !important;
  opacity: 1 !important;
}

.stTextInput [data-testid="InputInstructions"] {
  color: #475569 !important;
}

.stTextInput input:focus {
  border-color: #3b82f6 !important;
  box-shadow: 0 0 0 2px rgba(59,130,246,0.2), 0 0 20px rgba(59,130,246,0.1) !important;
}

.stButton > button,
.stFormSubmitButton > button {
  background: linear-gradient(135deg, #1e293b, #1a1d2e) !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 10px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 8px 20px !important;
  transition: all 0.2s ease !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
  border-color: rgba(59,130,246,0.4) !important;
  background: linear-gradient(135deg, #252d3d, #1e2538) !important;
  box-shadow: 0 0 12px rgba(59,130,246,0.15) !important;
}

.stButton > button,
.stFormSubmitButton > button {
  min-width: 80px !important;
  white-space: nowrap !important;
}

[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  border: 1px solid rgba(59,130,246,0.3) !important;
  color: #3b82f6 !important;
  font-weight: 500 !important;
  width: 100% !important;
  padding: 10px 20px !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(59,130,246,0.1) !important;
  border-color: rgba(59,130,246,0.6) !important;
}

.answer-card {
  background: linear-gradient(135deg, rgba(26,29,46,0.8), rgba(26,29,46,0.4));
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 24px 28px;
  margin: 16px 0;
  font-size: 14px;
  line-height: 1.8;
  color: #e2e8f0;
  animation: fadeIn 0.5s ease-out;
}

.source-block {
  background: rgba(26,29,46,0.6);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 10px;
  padding: 14px 18px;
  font-size: 12px;
  color: #94a3b8;
  margin-top: 8px;
  font-family: 'JetBrains Mono', monospace;
  white-space: pre-wrap;
}

.response-meta {
  font-size: 11px;
  color: #475569;
  margin-top: 10px;
  text-align: right;
  font-family: 'JetBrains Mono', monospace;
}

.offline-banner {
  background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(245,158,11,0.03));
  border: 1px solid rgba(245,158,11,0.2);
  border-radius: 10px;
  padding: 12px 18px;
  color: #fbbf24;
  font-size: 13px;
  margin-bottom: 16px;
  font-family: 'Inter', sans-serif;
}

.sidebar-label {
  font-size: 10px;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-weight: 600;
  margin-bottom: 10px;
  font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  top: 15%;
  left: 55%;
  width: 700px;
  height: 700px;
  background: radial-gradient(circle, rgba(59,130,246,0.04) 0%, rgba(6,182,212,0.02) 40%, transparent 70%);
  border-radius: 50%;
  transform: translateX(-50%);
  pointer-events: none;
  z-index: 0;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeout {
  0% { opacity: 1; }
  70% { opacity: 1; }
  100% { opacity: 0; }
}

@keyframes pulse {
  0%, 100% { box-shadow: 0 0 8px rgba(6,182,212,0.3); }
  50% { box-shadow: 0 0 20px rgba(6,182,212,0.5); }
}

@keyframes bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30% { transform: translateY(-6px); opacity: 1; }
}

@keyframes statusPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}


div[data-testid="stStatusWidget"],
div[data-testid="stSpinner"] {
  background-color: #1a1d2e !important;
  color: #94a3b8 !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  border-radius: 10px !important;
}

div[data-testid="stSpinner"] > div {
  color: #94a3b8 !important;
}

.stAlert, [data-testid="stNotification"] {
  background-color: #1a1d2e !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

@media (max-width: 768px) {
  .block-container {
    padding: 1rem 0.75rem !important;
  }
  .answer-card {
    padding: 16px 18px;
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
    font-size: 16px !important;
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
  [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
  }
  [data-testid="stForm"] [data-testid="stHorizontalBlock"] > div {
    width: 100% !important;
    flex: 1 1 100% !important;
  }
}

/* suggested question buttons */
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) button[kind="secondary"] {
  background: rgba(26,29,46,0.6) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  border-radius: 20px !important;
  padding: 8px 16px !important;
  font-size: 12px !important;
  color: #94a3b8 !important;
  font-family: 'Inter', sans-serif !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  white-space: normal !important;
  height: auto !important;
  min-height: 40px !important;
}
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) button[kind="secondary"]:hover {
  background: rgba(59,130,246,0.15) !important;
  border-color: rgba(59,130,246,0.4) !important;
  color: #e2e8f0 !important;
}
</style>""", unsafe_allow_html=True)


def render_sidebar():
    """Sidebar: document list grouped by category with icons, model status, chunk count."""
    with st.sidebar:
        st.markdown('<div class="sidebar-label">documents</div>', unsafe_allow_html=True)

        pdfs = get_pdf_list()
        if pdfs:
            groups = {}
            for pdf in pdfs:
                parts = pdf.replace("\\", "/").split("/")
                if len(parts) > 1:
                    category = parts[0]
                    filename = parts[-1]
                else:
                    category = "other"
                    filename = parts[0]
                name = filename.replace(".pdf", "").replace("_", " ")
                groups.setdefault(category, []).append(name)

            for category, names in sorted(groups.items()):
                style = CATEGORY_STYLES.get(category, {"icon": "\U0001f4c4", "color": "#64748b", "bg": "rgba(100,116,139,0.1)"})
                label = category.replace("_", " ")
                with st.expander(f'{style["icon"]}  {label} ({len(names)})'):
                    for name in names:
                        st.markdown(
                            f'<div style="color:#cbd5e1; font-size:12px; font-family:Inter,sans-serif; '
                            f'padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.03);">'
                            f'<span style="color:{style["color"]}; margin-right:6px;">\u25b8</span>{name}</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(
                '<div style="color:#475569; font-size:12px;">no PDFs in data/</div>',
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

        # status bar at the bottom
        st.markdown("<br><br>", unsafe_allow_html=True)

        ollama_online = check_ollama()
        chunk_count = st.session_state.get("chunk_count", get_indexed_count())
        st.session_state["chunk_count"] = chunk_count

        if ollama_online:
            status_dot = '<span style="display:inline-block; width:6px; height:6px; background:#10b981; border-radius:50%; animation:statusPulse 2s ease-in-out infinite; margin-right:6px;"></span>'
            status_label = "online"
        else:
            status_dot = '<span style="display:inline-block; width:6px; height:6px; background:#ef4444; border-radius:50%; margin-right:6px;"></span>'
            status_label = "offline"

        st.markdown(
            f'<div style="background:rgba(26,29,46,0.6); border:1px solid rgba(255,255,255,0.05); '
            f'border-radius:10px; padding:12px 14px; font-family:JetBrains Mono,monospace; font-size:11px;">'
            f'<div style="color:#94a3b8; margin-bottom:6px;">{status_dot}{LLM_MODEL} &middot; {status_label}</div>'
            f'<div style="color:#475569;">{chunk_count} chunks indexed</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_answer(result):
    """Display an answer card with source cards and metadata."""
    answer = result["answer"]
    sources = result["sources"]
    elapsed = result["elapsed"]

    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

    if sources:
        cards_html = '<div style="display:flex; gap:10px; overflow-x:auto; padding:8px 0; margin-top:12px;">'
        for i, src in enumerate(sources):
            page_display = src["page"] if src["page"] != "?" else "n/a"
            short_name = src["filename"].replace(".pdf", "").split("\\")[-1].split("/")[-1]
            if len(short_name) > 30:
                short_name = short_name[:27] + "..."
            cards_html += (
                f'<div style="flex-shrink:0; background:linear-gradient(135deg,rgba(26,29,46,0.8),rgba(26,29,46,0.4)); '
                f'backdrop-filter:blur(8px); border:1px solid rgba(255,255,255,0.06); '
                f'border-radius:10px; padding:12px 16px; min-width:180px; max-width:220px;">'
                f'<div style="font-size:10px; color:#3b82f6; font-family:JetBrains Mono,monospace; margin-bottom:6px; font-weight:500;">[{i+1}]</div>'
                f'<div style="font-size:11px; color:#cbd5e1; font-family:Inter,sans-serif; line-height:1.4;">{short_name}</div>'
                f'<div style="font-size:10px; color:#475569; font-family:JetBrains Mono,monospace; margin-top:6px;">p. {page_display}</div>'
                f'</div>'
            )
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        with st.expander("full excerpts"):
            for src in sources:
                page_display = src["page"] if src["page"] != "?" else "n/a"
                st.markdown(
                    f'<div class="source-block">'
                    f'\u203a {src["filename"]}  p.{page_display}\n'
                    f'"{src["excerpt"]}"'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    source_count = len(sources) if sources else 0
    st.markdown(
        f'<div class="response-meta">'
        f'model: {LLM_MODEL} &nbsp;&middot;&nbsp; sources: {source_count} &nbsp;&middot;&nbsp; {elapsed}s'
        f'</div>',
        unsafe_allow_html=True,
    )


def process_question(question, vs):
    """Run the full RAG pipeline with thinking animation, pipeline steps, and streaming."""
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
    try:
        with stream_placeholder.container():
            streamed = st.write_stream(stream_answer(question, context_docs))
        collected_tokens.append(streamed)
    except Exception:
        collected_tokens.append(
            "I could not find this information in the indexed documents."
        )

    # step 4: answer
    _render_pipeline_done(pipeline_placeholder)
    time.sleep(0.5)
    pipeline_placeholder.empty()

    wheel_placeholder.markdown(WHEEL_DONE, unsafe_allow_html=True)
    time.sleep(1.0)
    wheel_placeholder.empty()

    # clear streamed output -- re-rendered as answer card on rerun
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

    # header with title + github icon
    st.markdown(
        '<div style="display:flex; align-items:center; justify-content:space-between; '
        'border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:16px; margin-bottom:24px;">'
        '<div style="display:flex; align-items:center; gap:12px;">'
        '<div>'
        '<h2 style="color:#f1f5f9; font-family:Inter,sans-serif; font-weight:600; margin:0; font-size:20px;">'
        'Automotive RAG Assistant</h2>'
        '<span style="color:#475569; font-size:11px; font-family:JetBrains Mono,monospace;">'
        'query your engineering docs with AI</span>'
        '</div></div>'
        '<a href="https://github.com/jeanmira/automotive-rag-assistant" target="_blank" '
        'style="color:#475569; text-decoration:none; padding:8px;">'
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">'
        '<path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 '
        '0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695'
        '-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99'
        '.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225'
        '-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405'
        'c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 '
        '4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 '
        '.315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>'
        '</svg></a></div>',
        unsafe_allow_html=True,
    )

    if not check_ollama():
        st.markdown(
            '<div class="offline-banner">'
            '\u26a0 &nbsp;ollama is not running &mdash; start it with: <code>ollama serve</code>'
            '</div>',
            unsafe_allow_html=True,
        )

    # display chat history
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div style="color:#94a3b8; font-size:13px; font-family:Inter,sans-serif; '
                f'margin:20px 0 6px 0; display:flex; align-items:center; gap:8px;">'
                f'<span style="color:#3b82f6;">\u276f</span> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            render_answer(msg["result"])

    # suggested questions -- only show when chat is empty
    if not st.session_state["messages"]:
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            with cols[i]:
                if st.button(q, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state["_suggested_question"] = q
                    st.rerun()

    # input form
    with st.form("question_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([7, 0.6, 0.6], gap="small")
        with col1:
            user_input = st.text_input(
                "question",
                placeholder="ask about the docs...",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("ask")
        with col3:
            cleared = st.form_submit_button("clear")

    if cleared:
        st.session_state["messages"] = []
        st.rerun()

    question = None
    if submitted and user_input and user_input.strip():
        question = user_input.strip()

    # handle suggested question click
    if "_suggested_question" in st.session_state:
        question = st.session_state.pop("_suggested_question")

    if question:

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
                'no documents indexed &mdash; click "index documents" in the sidebar first'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        st.session_state["messages"].append({"role": "user", "content": question})

        try:
            vs = st.session_state.get("vector_store")
            if vs is None:
                vs = _cached_vector_store()
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
