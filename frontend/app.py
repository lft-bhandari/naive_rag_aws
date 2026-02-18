"""
RAG Microservice – Streamlit Frontend
Provides a document upload widget and an interactive chat interface.
"""

import os
import time
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://backend:8000")

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main layout */
    .main { background: #0f1117; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #161b27;
        border-right: 1px solid #2d3347;
    }

    /* Header */
    .rag-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .rag-header h1 { color: #fff; font-size: 2rem; margin: 0; }
    .rag-header p  { color: rgba(255,255,255,0.8); margin: 0.25rem 0 0; }

    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff;
        padding: 0.9rem 1.2rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 78%;
        width: fit-content;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .chat-assistant {
        background: #1e2535;
        color: #e2e8f0;
        padding: 0.9rem 1.2rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 78%;
        width: fit-content;
        font-size: 0.95rem;
        line-height: 1.5;
        border: 1px solid #2d3347;
    }

    /* Source card */
    .source-card {
        background: #0d1117;
        border: 1px solid #2d3347;
        border-left: 3px solid #667eea;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin: 0.3rem 0;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .source-score { color: #667eea; font-weight: 600; }

    /* Status badge */
    .status-ok  { color: #4ade80; font-weight: 600; }
    .status-err { color: #f87171; font-weight: 600; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2d3347 !important;
        border-radius: 10px !important;
        background: #161b27 !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton button:hover { opacity: 0.85; }

    /* Input */
    .stTextInput input, .stChatInput textarea {
        background: #161b27 !important;
        border: 1px solid #2d3347 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_index(file_bytes: bytes, filename: str) -> dict:
    r = requests.post(
        f"{API_BASE}/index",
        files={"file": (filename, file_bytes, "application/octet-stream")},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def api_chat(query: str, top_k: int, max_tokens: int) -> dict:
    r = requests.post(
        f"{API_BASE}/chat",
        json={"query": query, "top_k": top_k, "max_new_tokens": max_tokens},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def render_message(role: str, content: str, sources: list | None = None):
    if role == "user":
        st.markdown(f'<div class="chat-user">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-assistant">{content}</div>', unsafe_allow_html=True)
        if sources:
            with st.expander(f"📚 {len(sources)} source chunk(s) used", expanded=False):
                for s in sources:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="source-score">Score: {s["score"]}</span> · '
                        f'<strong>{s["source"]}</strong> · chunk {s["chunk_id"]}<br>'
                        f'{s["text"][:200]}…'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content, sources?}
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    top_k = st.slider("Retrieval top-k", 1, 10, 5,
                       help="Number of chunks retrieved per query")
    max_tokens = st.slider("Max response tokens", 128, 1024, 512, step=64,
                           help="Maximum tokens the LLM will generate")

    st.markdown("---")

    # Backend health
    st.markdown("### 🔌 Backend Status")
    health = api_health()
    if health:
        st.markdown(f'<span class="status-ok">● Connected</span>', unsafe_allow_html=True)
        st.caption(f"Device: `{health.get('device', '?')}`")
        st.caption(f"Collection: `{health.get('collection', '?')}`")
    else:
        st.markdown('<span class="status-err">● Unreachable</span>', unsafe_allow_html=True)
        st.caption(f"Endpoint: `{API_BASE}`")

    st.markdown("---")

    # Document upload
    st.markdown("### 📄 Upload Documents")
    uploaded = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and st.button("Index Selected Files", use_container_width=True):
        for uf in uploaded:
            if uf.name in st.session_state.indexed_docs:
                st.info(f"'{uf.name}' already indexed – skipping.")
                continue
            with st.spinner(f"Indexing {uf.name}…"):
                try:
                    result = api_index(uf.read(), uf.name)
                    st.session_state.indexed_docs.append(uf.name)
                    st.success(
                        f"✅ **{uf.name}** – {result['chunks_indexed']} chunks indexed"
                    )
                except Exception as e:
                    st.error(f"❌ Failed to index {uf.name}: {e}")

    if st.session_state.indexed_docs:
        st.markdown("**Indexed:**")
        for doc in st.session_state.indexed_docs:
            st.markdown(f"- 📄 {doc}")

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <h1>🔍 RAG Assistant</h1>
    <p>Upload documents, then ask questions grounded in their content</p>
</div>
""", unsafe_allow_html=True)

# Chat history display
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center; color:#4a5568; padding: 3rem 1rem;'>
            <div style='font-size:3rem'>💬</div>
            <h3 style='color:#667eea;'>Start a conversation</h3>
            <p>Upload documents via the sidebar, then ask anything about them.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_message(msg["role"], msg["content"], msg.get("sources"))

# Chat input
query = st.chat_input("Ask a question about your documents…")

if query:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})

    if not health:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Backend is not reachable. Please check the service.",
        })
    else:
        with st.spinner("Thinking…"):
            try:
                result = api_chat(query, top_k, max_tokens)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {e}",
                })

    st.rerun()