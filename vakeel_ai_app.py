# =============================================================
# vakeel_ai_app.py — Vakeel.AI Streamlit UI
# Run: streamlit run vakeel_ai_app.py
# =============================================================

import streamlit as st
import uuid
import time
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ───────────────
st.set_page_config(
    page_title="Vakeel.AI — Indian Law Intelligence",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# PREMIUM DARK THEME — Navy, Gold, Glassmorphism
# Completely different from friend's plain default Streamlit
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* ── Root Variables ───────────────────────────────────── */
:root {
    --navy-900: #050d1a;
    --navy-800: #0b1829;
    --navy-700: #0f2040;
    --navy-600: #162850;
    --gold-400: #d4a843;
    --gold-300: #e8c060;
    --gold-200: #f5d98a;
    --teal-400: #2dd4bf;
    --text-primary: #e8eaf0;
    --text-secondary: #8fa0b4;
    --glass-bg: rgba(15, 32, 64, 0.7);
    --glass-border: rgba(212, 168, 67, 0.2);
    --radius: 14px;
}

/* ── App Background ───────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #050d1a 0%, #0b1829 40%, #0f2040 100%);
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit default chrome ───────────────────── */
#MainMenu, header, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main container ───────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 960px;
}

/* ── Hero header ──────────────────────────────────────── */
.vakeel-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    background: linear-gradient(180deg, rgba(212,168,67,0.08) 0%, transparent 100%);
    border-bottom: 1px solid var(--glass-border);
    margin-bottom: 1.5rem;
    border-radius: var(--radius);
}
.vakeel-logo {
    font-size: 3.2rem;
    margin-bottom: 0.2rem;
}
.vakeel-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #d4a843 0%, #f5d98a 50%, #d4a843 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.vakeel-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
}
.vakeel-badge {
    display: inline-block;
    background: rgba(212,168,67,0.15);
    border: 1px solid rgba(212,168,67,0.4);
    color: var(--gold-300);
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.6rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--navy-800) !important;
    border-right: 1px solid var(--glass-border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--gold-300) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: var(--text-primary) !important;
    font-size: 0.875rem;
}

/* ── Sidebar section cards ────────────────────────────── */
.sidebar-card {
    background: rgba(212,168,67,0.07);
    border: 1px solid var(--glass-border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}
.sidebar-badge {
    display: inline-block;
    background: rgba(45, 212, 191, 0.12);
    border: 1px solid rgba(45,212,191,0.3);
    color: #2dd4bf;
    padding: 0.15rem 0.6rem;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* ── Chat message bubbles ─────────────────────────────── */
[data-testid="stChatMessageContent"] {
    background: transparent !important;
}
.user-bubble {
    background: linear-gradient(135deg, rgba(212,168,67,0.18), rgba(212,168,67,0.08));
    border: 1px solid rgba(212,168,67,0.3);
    border-radius: 14px 14px 4px 14px;
    padding: 0.9rem 1.1rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    font-size: 0.93rem;
    line-height: 1.6;
}
.assistant-bubble {
    background: rgba(15, 32, 64, 0.8);
    border: 1px solid rgba(212,168,67,0.15);
    border-radius: 14px 14px 14px 4px;
    padding: 0.9rem 1.1rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    font-size: 0.93rem;
    line-height: 1.6;
    backdrop-filter: blur(10px);
}

/* ── Metadata badge row ───────────────────────────────── */
.meta-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.6rem;
    align-items: center;
}
.meta-badge {
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.4px;
}
.badge-route {
    background: rgba(45,212,191,0.12);
    border: 1px solid rgba(45,212,191,0.35);
    color: #2dd4bf;
}
.badge-faith-good {
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.35);
    color: #4ade80;
}
.badge-faith-warn {
    background: rgba(234,179,8,0.12);
    border: 1px solid rgba(234,179,8,0.35);
    color: #facc15;
}
.badge-source {
    background: rgba(212,168,67,0.08);
    border: 1px solid rgba(212,168,67,0.25);
    color: var(--gold-300);
}

/* ── Chat input ───────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: var(--navy-700) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(212,168,67,0.6) !important;
    box-shadow: 0 0 0 2px rgba(212,168,67,0.12) !important;
}

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #d4a843, #c8983a) !important;
    color: #050d1a !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e8c060, #d4a843) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(212,168,67,0.3) !important;
}

/* ── Dividers ─────────────────────────────────────────── */
hr {
    border-color: var(--glass-border) !important;
}

/* ── Metric cards ─────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem !important;
}
[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--gold-300) !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}

/* ── Spinner ──────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--gold-400) !important;
}

/* ── Success / error banners ──────────────────────────── */
.stSuccess {
    background: rgba(34,197,94,0.1) !important;
    border: 1px solid rgba(34,197,94,0.3) !important;
    border-radius: 10px !important;
    color: #4ade80 !important;
}
.stError {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 10px !important;
}

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy-900); }
::-webkit-scrollbar-thumb { background: var(--navy-600); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold-400); }

/* ── Typing indicator animation ───────────────────────── */
@keyframes pulse-dot {
    0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
    40%            { transform: scale(1);   opacity: 1; }
}
.typing-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--gold-400);
    border-radius: 50%;
    margin: 0 2px;
    animation: pulse-dot 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# LOAD AGENT (cached — only runs once per session)
# ═══════════════════════════════════════════════════════════
from vakeel_ai_agent import build_agent, DOCUMENTS, IPC_LOOKUP

@st.cache_resource(show_spinner=False)
def get_agent():
    return build_agent()


# ═══════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════
if "messages"    not in st.session_state: st.session_state.messages    = []
if "thread_id"   not in st.session_state: st.session_state.thread_id   = str(uuid.uuid4())[:8]
if "last_meta"   not in st.session_state: st.session_state.last_meta   = {}
if "turn_count"  not in st.session_state: st.session_state.turn_count  = 0
if "agent_ready" not in st.session_state: st.session_state.agent_ready = False


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2rem;">⚖</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.3rem;
                    background:linear-gradient(90deg,#d4a843,#f5d98a);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; font-weight:700;">Vakeel.AI</div>
        <div style="color:#8fa0b4; font-size:0.75rem; margin-top:0.2rem;">Indian Law Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Knowledge base
    st.markdown('<div class="sidebar-badge">📚 KNOWLEDGE BASE</div>', unsafe_allow_html=True)
    for d in DOCUMENTS:
        topic_short = d["topic"][:48] + ("…" if len(d["topic"]) > 48 else "")
        st.markdown(f"<p style='margin:2px 0; font-size:0.8rem; color:#8fa0b4;'>• {topic_short}</p>",
                    unsafe_allow_html=True)

    st.divider()

    # Agent capabilities
    st.markdown('<div class="sidebar-badge">🤖 AGENT CAPABILITIES</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-card">
        <p style="color:#e8eaf0; margin:0; font-size:0.82rem; line-height:1.7;">
        ✅ LangGraph StateGraph (9 nodes)<br>
        ✅ ChromaDB RAG (10 Indian law docs)<br>
        ✅ MemorySaver + thread_id memory<br>
        ✅ Self-reflection eval node<br>
        ✅ IPC Section Lookup Tool<br>
        ✅ Clarify Node for vague queries<br>
        ✅ Streamlit deployment
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Session info
    st.markdown('<div class="sidebar-badge">🔖 SESSION</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Thread", f"#{st.session_state.thread_id}")
    col2.metric("Turns", st.session_state.turn_count)

    # Last response metadata
    if st.session_state.last_meta:
        st.markdown('<div class="sidebar-badge" style="margin-top:0.75rem;">📊 LAST RESPONSE</div>', unsafe_allow_html=True)
        meta = st.session_state.last_meta
        route = meta.get("route", "—")
        faith = meta.get("faithfulness", 0.0)
        sources = meta.get("sources", [])

        faith_color = "#4ade80" if faith >= 0.65 else "#facc15"
        st.markdown(f"""
        <div class="sidebar-card">
            <p style="color:#8fa0b4; font-size:0.75rem; margin:0 0 0.3rem;">Route</p>
            <p style="color:#2dd4bf; font-weight:600; font-size:0.9rem; margin:0 0 0.6rem;">{route.upper()}</p>
            <p style="color:#8fa0b4; font-size:0.75rem; margin:0 0 0.3rem;">Faithfulness</p>
            <p style="color:{faith_color}; font-weight:700; font-size:1.1rem; margin:0;">
                {'●' if faith >= 0.65 else '◐'} {faith:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        if sources:
            st.markdown('<p style="color:#8fa0b4; font-size:0.75rem; margin:0.4rem 0 0.2rem;">Sources cited:</p>', unsafe_allow_html=True)
            for s in sources:
                s_short = s[:42] + "…" if len(s) > 42 else s
                st.markdown(f"<p style='margin:1px 0; color:#d4a843; font-size:0.76rem;'>📄 {s_short}</p>",
                            unsafe_allow_html=True)

    st.divider()

    # IPC Quick Lookup
    st.markdown('<div class="sidebar-badge">⚡ QUICK IPC LOOKUP</div>', unsafe_allow_html=True)
    crime_input = st.selectbox(
        "Select offence:",
        options=[""] + list(IPC_LOOKUP.keys()),
        format_func=lambda x: x.title() if x else "Choose an offence…",
        label_visibility="collapsed",
    )
    if crime_input:
        section, punishment = IPC_LOOKUP[crime_input]
        st.markdown(f"""
        <div class="sidebar-card">
            <p style="color:#d4a843; font-weight:600; font-size:0.85rem; margin:0 0 0.3rem;">{section}</p>
            <p style="color:#8fa0b4; font-size:0.8rem; margin:0;">{punishment}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 New Conversation"):
        st.session_state.messages   = []
        st.session_state.thread_id  = str(uuid.uuid4())[:8]
        st.session_state.last_meta  = {}
        st.session_state.turn_count = 0
        st.rerun()


# ═══════════════════════════════════════════════════════════
# MAIN AREA — Hero Header
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="vakeel-header">
    <div class="vakeel-logo">⚖</div>
    <h1 class="vakeel-title">Vakeel.AI</h1>
    <p class="vakeel-subtitle">Indian Law Intelligence Assistant &nbsp;|&nbsp; Powered by LangGraph + Groq</p>
    <span class="vakeel-badge">🇮🇳 Indian Law · IPC · Constitution · Consumer · RTI · POCSO · CrPC · IT Act</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# LOAD AGENT
# ═══════════════════════════════════════════════════════════
with st.spinner("⚖ Loading Indian Law Knowledge Base..."):
    try:
        agent_app, embedder, collection = get_agent()
        st.session_state.agent_ready = True
    except Exception as e:
        st.error(f"❌ Failed to load Vakeel.AI: {e}")
        st.info("Make sure GROQ_API_KEY is set in your .env file.")
        st.stop()

if st.session_state.agent_ready and st.session_state.turn_count == 0:
    st.success(f"✅ Knowledge base ready — {collection.count()} Indian law documents indexed")


# ═══════════════════════════════════════════════════════════
# SUGGESTED STARTER QUESTIONS
# ═══════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown("""
    <p style="color:#8fa0b4; font-size:0.85rem; text-align:center; margin-bottom:0.75rem; margin-top:0.5rem;">
        💡 Try one of these questions to get started
    </p>
    """, unsafe_allow_html=True)

    starters = [
        "What are my fundamental rights under the Indian Constitution?",
        "How do I file an RTI application?",
        "Which IPC section applies to cybercrime?",
        "How to file a consumer complaint in India?",
    ]
    cols = st.columns(2)
    for i, s in enumerate(starters):
        if cols[i % 2].button(f"💬 {s}", key=f"starter_{i}"):
            st.session_state._starter_q = s
            st.rerun()

    # Handle starter click
    if hasattr(st.session_state, "_starter_q"):
        starter = st.session_state._starter_q
        del st.session_state._starter_q
        st.session_state.messages.append({"role": "user", "content": starter})
        st.session_state.turn_count += 1

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = agent_app.invoke({"question": starter}, config=config)
        answer = result.get("answer", "Sorry, I could not generate a response.")
        route  = result.get("route", "")
        faith  = result.get("faithfulness", 0.0)
        sources = result.get("sources", [])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_meta = {"route": route, "faithfulness": faith, "sources": sources}
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# CHAT HISTORY DISPLAY
# ═══════════════════════════════════════════════════════════
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(
                f'<div class="user-bubble">🧑‍💼 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="assistant-bubble">⚖ {msg["content"]}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════
# CHAT INPUT
# ═══════════════════════════════════════════════════════════
if user_input := st.chat_input("Ask about Indian law, your rights, IPC sections, filing complaints..."):

    # Display user message
    with st.chat_message("user"):
        st.markdown(
            f'<div class="user-bubble">🧑‍💼 {user_input}</div>',
            unsafe_allow_html=True,
        )
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Invoke agent
    with st.chat_message("assistant"):
        with st.spinner("⚖ Vakeel.AI is researching Indian law..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": user_input}, config=config)

        answer  = result.get("answer",      "Sorry, I could not generate a response.")
        route   = result.get("route",       "retrieve")
        faith   = result.get("faithfulness", 0.0)
        sources = result.get("sources",     [])

        # Answer bubble
        st.markdown(
            f'<div class="assistant-bubble">⚖ {answer}</div>',
            unsafe_allow_html=True,
        )

        # Metadata badges
        faith_cls = "badge-faith-good" if faith >= 0.65 else "badge-faith-warn"
        faith_icon = "🟢" if faith >= 0.65 else "🟡"

        badges_html = f'<div class="meta-row">'
        badges_html += f'<span class="meta-badge badge-route">⚡ {route.upper()}</span>'
        if faith > 0:
            badges_html += f'<span class="meta-badge {faith_cls}">{faith_icon} Faithfulness: {faith:.2f}</span>'
        for s in sources[:2]:
            s_short = s[:35] + "…" if len(s) > 35 else s
            badges_html += f'<span class="meta-badge badge-source">📄 {s_short}</span>'
        badges_html += '</div>'
        st.markdown(badges_html, unsafe_allow_html=True)

    # Update session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_meta  = {"route": route, "faithfulness": faith, "sources": sources}
    st.session_state.turn_count += 1


# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 2rem 0 0.5rem; color:#4a5568; font-size:0.75rem;">
    ⚖ Vakeel.AI — For educational purposes only. Not a substitute for professional legal advice.<br>
    Built with LangGraph · ChromaDB · Groq · Sentence-Transformers
</div>
""", unsafe_allow_html=True)
