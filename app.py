"""
app.py
------
Streamlit interface for querying the ML papers corpus via RAG.
100% free stack: Streamlit (open-source), HuggingFace embeddings (local),
Chroma (local), Groq (free LLM with a free API key).

Includes a simple email/password authentication gate (see auth.py).

Run with:
    streamlit run app.py
"""

import os
import streamlit as st

import rag_core as rc
import auth

st.set_page_config(
    page_title="ML Papers RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# App-wide Groq API key — configured once by the app owner,
# never entered by end users.
#
# Priority:
#   1. Streamlit secrets  -> .streamlit/secrets.toml (local) or
#      the "Secrets" panel in Streamlit Community Cloud (deployed).
#   2. Environment variable GROQ_API_KEY (e.g. set in your shell,
#      a .env file loaded before `streamlit run`, or your host's
#      environment variable settings).
# =========================================================
def get_groq_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        # st.secrets raises if no secrets.toml file exists at all — that's fine,
        # it just means we fall back to the environment variable below.
        pass
    return os.environ.get("GROQ_API_KEY", "")


GROQ_API_KEY = get_groq_api_key()

# =========================================================
# Global styling — professional look & feel
# =========================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        #MainMenu, footer, header {visibility: hidden;}

        :root {
            --brand-primary: #4F46E5;
            --brand-primary-dark: #3730A3;
            --brand-bg: #F8F9FC;
        }

        .stApp {
            background-color: var(--brand-bg);
        }

        .brand-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding-bottom: 0.25rem;
        }
        .brand-title {
            font-weight: 700;
            font-size: 1.5rem;
            color: #1E1B4B;
        }
        .brand-subtitle {
            color: #6B7280;
            font-size: 0.92rem;
        }

        .auth-card {
            max-width: 420px;
            margin: 3.5rem auto 0 auto;
            padding: 2.25rem 2.25rem 1.75rem 2.25rem;
            background: white;
            border-radius: 16px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 4px 24px rgba(17, 24, 39, 0.06);
        }
        .auth-title {
            font-weight: 700;
            font-size: 1.35rem;
            color: #111827;
            margin-bottom: 0.15rem;
        }
        .auth-subtitle {
            color: #6B7280;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }

        div.stButton > button {
            border-radius: 8px;
            font-weight: 600;
            border: none;
            background-color: var(--brand-primary);
            color: white;
            transition: background-color 0.15s ease;
        }
        div.stButton > button:hover {
            background-color: var(--brand-primary-dark);
            color: white;
        }

        .source-card {
            padding: 0.85rem 1rem;
            background: white;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            margin-bottom: 0.6rem;
        }
        .source-score {
            display: inline-block;
            padding: 0.05rem 0.5rem;
            border-radius: 6px;
            background: #EEF2FF;
            color: var(--brand-primary-dark);
            font-size: 0.78rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Authentication gate
# =========================================================
def render_login_page():
    st.markdown(
        """
        <div class="brand-header" style="justify-content:center; margin-top:2rem;">
            <span style="font-size:2rem;">📚</span>
            <span class="brand-title">ML Papers RAG</span>
        </div>
        <div class="brand-subtitle" style="text-align:center;">
            Your research assistant for foundational machine learning papers
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Sign in", "Create account"])

    with tab_login:
        st.markdown('<div class="auth-title">Welcome back</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Sign in to continue</div>', unsafe_allow_html=True)

        with st.form("login_form"):
            email = st.text_input("Email", key="login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please fill in both fields.")
            else:
                success, name, message = auth.verify_user(email, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email.strip().lower()
                    st.session_state.user_name = name
                    st.rerun()
                else:
                    st.error(message)

    with tab_signup:
        st.markdown('<div class="auth-title">Create your account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">It only takes a minute</div>', unsafe_allow_html=True)

        with st.form("signup_form"):
            name = st.text_input("Full name", key="signup_name", placeholder="Jane Doe")
            email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
            password = st.text_input(
                "Password", type="password", key="signup_password",
                help="At least 8 characters.",
            )
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Create account", use_container_width=True)

        if submitted:
            if not name or not email or not password:
                st.error("Please fill in all fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = auth.create_user(email, name, password)
                if success:
                    st.success(message + " Please sign in from the other tab.")
                else:
                    st.error(message)

    st.markdown("</div>", unsafe_allow_html=True)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    render_login_page()
    st.stop()


# =========================================================
# Sidebar — configuration (only shown once logged in)
# =========================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="brand-header">
            <span style="font-size:1.6rem;">📚</span>
            <div>
                <div class="brand-title" style="font-size:1.15rem;">ML Papers RAG</div>
                <div class="brand-subtitle">Research assistant for 26 foundational ML papers</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(f"Signed in as **{st.session_state.get('user_name', '')}**")
    st.caption(st.session_state.get("user_email", ""))
    if st.button("Log out", use_container_width=True):
        for key in ("authenticated", "user_email", "user_name"):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()

    if GROQ_API_KEY:
        st.caption("🟢 Groq API connected")
    else:
        st.caption("🔴 Groq API key not configured")

    st.divider()
    top_k = st.slider("Number of sources to use", min_value=2, max_value=10, value=rc.FINAL_TOP_K)

    verify_answers = st.toggle(
        "Anti-hallucination check",
        value=True,
        help="Adds a second LLM pass that re-reads the answer and removes/corrects "
             "any claim not explicitly present in the sources. "
             "More reliable, but roughly 2x slower.",
    )

    st.divider()
    with st.expander("📄 Indexed papers"):
        for name in sorted(rc.PAPER_TITLES.values()):
            st.write(f"- {name}")

    st.divider()
    st.caption(
        "⚠️ On first launch, the app downloads and indexes 26 PDFs "
        "(~5-15 min depending on your connection). Subsequent launches are "
        "nearly instant thanks to local caching."
    )


# =========================================================
# Pipeline loading (cached — runs only once)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_pipeline():
    logs = []

    def log(msg):
        logs.append(msg)

    merged = rc.load_and_merge_documents(log=log)
    chunks = rc.build_chunks(merged, log=log)
    vectorstore = rc.build_vectorstore(chunks, log=log)
    reranker = rc.load_reranker(log=log)

    return vectorstore, reranker, logs


status_placeholder = st.empty()
with status_placeholder.container():
    with st.spinner("Preparing the corpus (downloading / indexing on first launch)..."):
        vectorstore, reranker, pipeline_logs = load_pipeline()
status_placeholder.empty()

with st.expander("🛠️ Pipeline preparation logs"):
    for line in pipeline_logs:
        st.text(line)


# =========================================================
# Chat area
# =========================================================
st.markdown(
    """
    <div class="brand-header">
        <span style="font-size:1.7rem;">💬</span>
        <span class="brand-title">Ask a question about the papers</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Transformer, BERT, GPT, LoRA, diffusion, RAG, RL... answers are "
    "generated exclusively from the indexed papers, with source citations."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} sources used"):
                for s in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <strong>{s['paper_name']}</strong>
                            <span class="source-score">relevance {s['score']:.3f}</span>
                            <div style="color:#6B7280; font-size:0.85rem; margin-top:0.4rem;">
                                {s['excerpt']}…
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

prompt = st.chat_input("E.g. How does the attention mechanism work in the Transformer?")

if prompt:
    if not GROQ_API_KEY:
        st.error(
            "⚠️ No Groq API key is configured for this app. The app owner needs to "
            "set GROQ_API_KEY in `.streamlit/secrets.toml` (or as an environment "
            "variable) — see the comment at the top of app.py."
        )
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the corpus and generating the answer..."):
                try:
                    llm = rc.load_llm(api_key=GROQ_API_KEY)
                    answer, sources = rc.generate_answer(
                        prompt, vectorstore, reranker, llm, top_k=top_k, verify=verify_answers
                    )
                except Exception as e:
                    answer = f"❌ Error while generating the answer: {e}"
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander(f"📎 {len(sources)} sources used"):
                    for s in sources:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <strong>{s['paper_name']}</strong>
                                <span class="source-score">relevance {s['score']:.3f}</span>
                                <div style="color:#6B7280; font-size:0.85rem; margin-top:0.4rem;">
                                    {s['excerpt']}…
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

if not st.session_state.messages:
    st.info(
        "💡 Try asking, for example: *\"What is LoRA and how does it reduce "
        "fine-tuning cost?\"* or *\"How do diffusion models work?\"*"
    )