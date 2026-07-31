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
import secrets
import streamlit as st

import rag_core as rc
import auth
import history

st.set_page_config(
    page_title="ML Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Lightweight session persistence
# ---------------------------------------------------------
# st.session_state is wiped on every full page reload/refresh — that's
# normal Streamlit behavior, not a bug, but it means a refresh always
# bounces the user back to the login page. To survive a refresh, we store
# a random session token in the URL's query params (which DOES survive a
# reload, since the browser keeps the same URL) and look it up server-side
# in an in-memory store to restore the login state.
#
# Note: this store lives in the app process's memory. If the Streamlit
# Cloud container restarts (sleep/redeploy), all active sessions are
# cleared and everyone has to log in again — but a normal page refresh no
# longer logs anyone out, which was the actual problem being fixed here.
# =========================================================
@st.cache_resource
def _session_store():
    return {}  # token -> {"email": ..., "name": ...}


def _create_session(email, name):
    token = secrets.token_urlsafe(24)
    _session_store()[token] = {"email": email, "name": name}
    return token


def _get_session(token):
    return _session_store().get(token)


def _destroy_session(token):
    _session_store().pop(token, None)


# =========================================================
# App-wide Groq API keys — configured once by the app owner,
# never entered by end users.
#
# Supports multiple keys (e.g. several free-tier accounts) so that if one
# key hits its daily rate limit, the app automatically falls back to the
# next one instead of failing. Add as many as you like in secrets.toml as
# GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ... (any number, any names
# following that pattern work).
# =========================================================
def get_groq_api_keys():
    keys = []

    # Collect from st.secrets (GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ...)
    try:
        for key_name in st.secrets.keys():
            if key_name == "GROQ_API_KEY" or key_name.startswith("GROQ_API_KEY_"):
                val = st.secrets[key_name]
                if val:
                    keys.append(val)
    except Exception:
        pass

    # Also collect from environment variables, same naming pattern
    for key_name in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4",
                      "GROQ_API_KEY_5"]:
        val = os.environ.get(key_name)
        if val:
            keys.append(val)

    # De-duplicate while preserving order
    seen = set()
    unique_keys = []
    for k in keys:
        if k and k not in seen:
            unique_keys.append(k)
            seen.add(k)

    return unique_keys


GROQ_API_KEYS = get_groq_api_keys()

# =========================================================
# Global styling — professional look & feel
# =========================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Force the LIGHT color scheme regardless of the device/browser's
           system dark mode setting. Without this, phones in dark mode make
           Chrome auto-invert/auto-darken form controls and default text
           colors on top of our own light-theme CSS, which is what was
           washing out (making invisible) the chat input and message text
           on mobile. */
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            color-scheme: light only;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        #MainMenu, footer {visibility: hidden;}

        :root {
            --brand-primary: #6B7280;
            --brand-primary-dark: #4B5563;
            --brand-bg: #F8F9FC;
            --border-color: #E5E7EB;
            --text-muted: #6B7280;
        }

        .stApp {
            background-color: var(--brand-bg);
        }

        /* Remove Streamlit's default extra bottom padding that leaves a
           dead empty gap below the last element */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
            max-width: 1100px;
        }

        /* Fix the default white bar behind the chat input so it blends
           with the app background instead of showing as a stray white
           rectangle */
        [data-testid="stChatInput"],
        [data-testid="stBottomBlockContainer"] {
            background: var(--brand-bg) !important;
            border-top: none !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] textarea {
            background: white !important;
            color: #111827 !important;
            border-radius: 12px !important;
            border: 1px solid var(--border-color) !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #6B7280 !important;
            opacity: 1 !important;
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
            color: var(--text-muted);
            font-size: 0.92rem;
        }

        /* Auth card — smaller, tighter, centered */
        .auth-card {
            max-width: 340px;
            margin: 1.5rem auto 0 auto;
            padding: 1.5rem 1.5rem 1.25rem 1.5rem;
            background: white;
            border-radius: 14px;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 20px rgba(17, 24, 39, 0.06);
        }
        .auth-title {
            font-weight: 700;
            font-size: 1.15rem;
            color: #111827;
            margin-bottom: 0.1rem;
        }
        .auth-subtitle {
            color: var(--text-muted);
            font-size: 0.82rem;
            margin-bottom: 1rem;
        }

        /* Tighter form field spacing inside the auth card */
        .auth-card [data-testid="stTextInput"] {
            margin-bottom: -0.6rem;
        }
        .auth-card [data-testid="stTextInput"] input {
            background: white !important;
            color: #111827 !important;
            border-radius: 8px !important;
            padding: 0.45rem 0.7rem !important;
            font-size: 0.9rem !important;
        }
        .auth-card [data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
        }

        div.stButton > button, div.stFormSubmitButton > button {
            border-radius: 8px;
            font-weight: 600;
            border: none;
            background-color: var(--brand-primary);
            color: white;
            transition: background-color 0.15s ease;
        }
        div.stButton > button:hover, div.stFormSubmitButton > button:hover {
            background-color: var(--brand-primary-dark);
            color: white;
        }

        /* Sidebar polish */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid var(--border-color);
        }

        /* Chat message bubbles */
        [data-testid="stChatMessage"] {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span {
            color: #111827 !important;
        }

        .source-card {
            padding: 0.85rem 1rem;
            background: #FAFAFC;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            margin-bottom: 0.6rem;
        }
        /* The bold "[1] Paper Name" line inside a source card had no
           explicit color, so it inherited Streamlit's own text-color
           variable — which the config.toml theme lock now keeps light,
           but this is a belt-and-suspenders explicit override too. */
        .source-card strong {
            color: #111827 !important;
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

        /* Expander polish */
        [data-testid="stExpander"] {
            border: 1px solid var(--border-color);
            border-radius: 10px;
            background: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Authentication gate
# =========================================================
def render_login_page():
    # The login page never uses st.chat_input, so Streamlit's reserved
    # bottom bar is just empty white space here — hide it completely.
    st.markdown(
        """
        <style>
            [data-testid="stBottomBlockContainer"],
            [data-testid="stBottom"] {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="brand-header" style="justify-content:center; margin-top:1rem;">
            <span style="font-size:1.8rem;">📚</span>
            <span class="brand-title">ML Research Assistant</span>
        </div>
        <div class="brand-subtitle" style="text-align:center; margin-bottom: 0.5rem;">
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
                    clean_email = email.strip().lower()
                    token = _create_session(clean_email, name)
                    st.session_state.authenticated = True
                    st.session_state.user_email = clean_email
                    st.session_state.user_name = name
                    st.session_state.session_token = token
                    st.query_params["session"] = token
                    st.rerun()
                else:
                    st.error(message)

        with st.expander("Forgot password?"):
            forgot_email = st.text_input(
                "Email", key="forgot_password_email", placeholder="you@example.com",
            )
            if st.button("Send reset link", key="send_reset_link_button"):
                if not forgot_email:
                    st.error("Please enter your email.")
                else:
                    app_url = st.secrets.get("APP_URL", "")
                    if not app_url:
                        st.error(
                            "APP_URL is not configured — the app owner needs to set it "
                            "in secrets.toml (e.g. https://yourapp.streamlit.app)."
                        )
                    else:
                        with st.spinner("Sending..."):
                            success, message = auth.request_password_reset(forgot_email, app_url)
                        if success:
                            st.success(message)
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


def render_reset_password_page(token):
    """Shown instead of the normal login page when the URL contains a
    ?reset_token=... param (i.e. the user clicked the link from their
    reset email)."""
    st.markdown(
        """
        <style>
            [data-testid="stBottomBlockContainer"],
            [data-testid="stBottom"] {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="brand-header" style="justify-content:center; margin-top:1rem;">
            <span style="font-size:1.8rem;">📚</span>
            <span class="brand-title">ML Research Assistant</span>
        </div>
        <div class="brand-subtitle" style="text-align:center; margin-bottom: 0.5rem;">
            Choose a new password
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    email = auth.verify_reset_token(token)

    if not email:
        st.error("This reset link is invalid or has expired. Please request a new one from the sign-in page.")
        if st.button("Back to sign in", use_container_width=True):
            st.query_params.clear()
            st.rerun()
    else:
        st.markdown(f'<div class="auth-title">Resetting password for</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="auth-subtitle">{email}</div>', unsafe_allow_html=True)

        with st.form("reset_password_form"):
            new_password = st.text_input(
                "New password", type="password", key="reset_new_password",
                help="At least 8 characters.",
            )
            confirm_password = st.text_input(
                "Confirm new password", type="password", key="reset_confirm_password",
            )
            submitted = st.form_submit_button("Reset password", use_container_width=True)

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = auth.reset_password(token, new_password)
                if success:
                    st.success(message)
                    if st.button("Go to sign in", use_container_width=True):
                        st.query_params.clear()
                        st.rerun()
                else:
                    st.error(message)

    st.markdown("</div>", unsafe_allow_html=True)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# If the URL carries a password-reset token (i.e. the user clicked the
# link from their reset email), show the reset-password page instead of
# anything else — this takes priority even if the browser happens to
# still have an active login session.
reset_token_from_url = st.query_params.get("reset_token")
if reset_token_from_url:
    render_reset_password_page(reset_token_from_url)
    st.stop()

# Try to restore the session from the URL's query param — this is what
# survives a page refresh, since session_state itself gets wiped but the
# browser keeps the same URL (and therefore the same "session" param).
if not st.session_state.authenticated:
    token_from_url = st.query_params.get("session")
    if token_from_url:
        session_data = _get_session(token_from_url)
        if session_data:
            st.session_state.authenticated = True
            st.session_state.user_email = session_data["email"]
            st.session_state.user_name = session_data["name"]
            st.session_state.session_token = token_from_url

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
                <div class="brand-title" style="font-size:1.15rem;">ML Research Assistant</div>
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
        token = st.session_state.get("session_token")
        if token:
            _destroy_session(token)
        st.query_params.clear()
        for key in ("authenticated", "user_email", "user_name", "session_token"):
            st.session_state.pop(key, None)
        st.rerun()

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
    st.markdown('<div class="brand-title" style="font-size:0.95rem;">🕘 Conversations</div>', unsafe_allow_html=True)

    with st.container(key="sidebar_history"):
        if st.button("➕ New conversation", use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

        past_conversations = history.list_conversations(st.session_state.user_email)
        for conv in past_conversations:
            is_active = conv["id"] == st.session_state.get("conversation_id")
            label = ("🟢 " if is_active else "") + conv["title"]

            col_load, col_delete = st.columns([5, 1])
            with col_load:
                if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                    st.session_state.conversation_id = conv["id"]
                    st.session_state.messages = history.load_messages(conv["id"])
                    st.rerun()
            with col_delete:
                if st.button("🗑", key=f"del_{conv['id']}", use_container_width=True):
                    history.delete_conversation(conv["id"])
                    if is_active:
                        st.session_state.conversation_id = None
                        st.session_state.messages = []
                    st.rerun()

    st.divider()
    with st.expander("📄 Indexed papers"):
        for name in sorted(rc.PAPER_TITLES.values()):
            st.write(f"- {name}")


# =========================================================
# Pipeline loading (cached — runs only once, no visible logs)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_pipeline():
    def log(msg):
        pass  # logs are intentionally discarded — no debug panel shown to users

    rc.ensure_prebuilt_assets(log=log)  # try the HF Hub prebuilt index first
    merged = rc.load_and_merge_documents(log=log)
    chunks = rc.build_chunks(merged, log=log)
    vectorstore = rc.build_vectorstore(chunks, log=log)
    reranker = rc.load_reranker(log=log)

    return vectorstore, reranker


with st.spinner("Preparing the corpus..."):
    vectorstore, reranker = load_pipeline()


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
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None


def render_feedback_buttons(msg):
    """Shows thumbs up/down for an assistant message, or a confirmation
    once feedback has already been given for it."""
    message_id = msg.get("id")
    if not message_id:
        return  # message wasn't persisted (e.g. an error message), nothing to attach feedback to

    current = msg.get("feedback")
    if current:
        label = "👍 Marked helpful" if current == "up" else "👎 Marked not helpful"
        st.caption(label)
        return

    col_up, col_down, _ = st.columns([1, 1, 8])
    with col_up:
        if st.button("👍", key=f"up_{message_id}"):
            history.set_message_feedback(message_id, "up")
            msg["feedback"] = "up"
            st.rerun()
    with col_down:
        if st.button("👎", key=f"down_{message_id}"):
            history.set_message_feedback(message_id, "down")
            msg["feedback"] = "down"
            st.rerun()


def render_sources(sources):
    with st.expander(f"📎 {len(sources)} sources used"):
        for s in sources:
            badge = f"[{s['number']}]" if s.get("number") else f"#{sources.index(s) + 1}"
            pages_html = (
                f'<div style="color:var(--brand-primary-dark); font-size:0.78rem; '
                f'margin-top:0.3rem; font-weight:600;">📄 Pages: {s["pages_display"]}</div>'
                if s.get("pages_display") else ""
            )
            st.markdown(
                f"""
                <div class="source-card">
                    <strong>{badge} {s['paper_name']}</strong>
                    {pages_html}
                    <div style="color:#6B7280; font-size:0.85rem; margin-top:0.4rem;">
                        {s['excerpt']}…
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg["role"] == "assistant":
            render_feedback_buttons(msg)

prompt = st.chat_input("E.g. How does the attention mechanism work in the Transformer?")

if prompt:
    if not GROQ_API_KEYS:
        st.error(
            "⚠️ No Groq API key is configured for this app. The app owner needs to "
            "set GROQ_API_KEY (and optionally GROQ_API_KEY_2, GROQ_API_KEY_3, ...) in "
            "`.streamlit/secrets.toml` (or as environment variables) — see the comment "
            "at the top of app.py."
        )
    else:
        # Start a new persisted conversation on the first message of a fresh chat
        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = history.create_conversation(
                st.session_state.user_email, title=prompt
            )

        user_message_id = history.save_message(
            st.session_state.conversation_id, "user", prompt
        )
        st.session_state.messages.append({
            "role": "user", "content": prompt, "id": user_message_id,
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the corpus and generating the answer..."):
                try:
                    answer, sources = rc.generate_answer(
                        prompt, vectorstore, reranker, GROQ_API_KEYS,
                        top_k=top_k, verify=verify_answers,
                    )
                except Exception as e:
                    answer = f"❌ Error while generating the answer: {e}"
                    sources = []

            st.markdown(answer)

            # Save first so we have a message id to attach feedback buttons to
            assistant_message_id = history.save_message(
                st.session_state.conversation_id, "assistant", answer, sources=sources
            )
            new_message = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "id": assistant_message_id,
            }
            st.session_state.messages.append(new_message)

            if sources:
                render_sources(sources)

            render_feedback_buttons(new_message)

if not st.session_state.messages:
    st.info(
        "💡 Try asking, for example: *\"What is LoRA and how does it reduce "
        "fine-tuning cost?\"* or *\"How do diffusion models work?\"*"
    )