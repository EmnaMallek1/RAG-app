"""
app.py
------
Streamlit interface for querying the ML papers corpus via RAG.
100% free stack: Streamlit (open-source), HuggingFace embeddings (local),
Chroma (local), Groq (free LLM with a free API key).

Includes a simple email/password authentication gate (see auth.py).

Theme:
    The whole app (sidebar, chat, sources, forms - everything) now supports
    two themes:
      - "dark"  (default): the starfield background, used app-wide, with
        light text so everything stays readable.
      - "light": the original plain light look.
    A small toggle at the very top of every page lets the user switch
    between them at any time; the choice is kept in st.session_state so it
    persists across reruns (but not across a full server restart, same as
    everything else in session_state).

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
# Theme state — default is DARK (starfield, app-wide).
# Switching to "light" restores the original plain look everywhere.
# =========================================================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

THEMES = {
    "light": {
        "bg": "#F8F9FC",
        "panel": "#FFFFFF",
        "panel_alt": "#FAFAFC",
        "border": "#E5E7EB",
        "text": "#111827",
        "text_muted": "#6B7280",
        "input_bg": "#FFFFFF",
        "input_text": "#111827",
        "placeholder": "#6B7280",
        "primary": "#6B7280",
        "primary_dark": "#4B5563",
        "badge_bg": "#EEF2FF",
        "badge_text": "#4B5563",
        "sidebar_bg": "#FFFFFF",
        "header_title": "#1E1B4B",
        "auth_card_bg": "#FFFFFF",
        "header_bg": "#FFFFFF",
        "stars_opacity": "0",
    },
    "dark": {
        "bg": "#0b1026",
        "panel": "rgba(255, 255, 255, 0.06)",
        "panel_alt": "rgba(255, 255, 255, 0.04)",
        "border": "rgba(255, 255, 255, 0.18)",
        "text": "#F8F9FC",
        "text_muted": "#C7CCE6",
        "input_bg": "rgba(255, 255, 255, 0.10)",
        "input_text": "#F8F9FC",
        "placeholder": "#AEB4D6",
        "primary": "#818CF8",
        "primary_dark": "#6366F1",
        "badge_bg": "rgba(129, 140, 248, 0.20)",
        "badge_text": "#E0E3FF",
        "sidebar_bg": "rgba(11, 16, 38, 0.88)",
        "header_title": "#F8F9FC",
        "auth_card_bg": "rgba(20, 26, 53, 0.85)",
        "header_bg": "#0b1026",
        "stars_opacity": "1",
    },
}

_theme = THEMES[st.session_state.theme_mode]

# =========================================================
# Global styling — built from the active theme's color palette so the
# ENTIRE app (sidebar, chat, sources, forms, everything) follows whichever
# mode is selected, not just the login screen.
# =========================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        #MainMenu, footer {visibility: hidden;}

        :root {
            --brand-primary: %(primary)s;
            --brand-primary-dark: %(primary_dark)s;
            --brand-bg: %(bg)s;
            --border-color: %(border)s;
            --text-muted: %(text_muted)s;
            --text-main: %(text)s;
            --panel-bg: %(panel)s;
            --panel-alt-bg: %(panel_alt)s;
        }

        .stApp {
            background-color: var(--brand-bg);
        }

        /* ---- FIX 1: the top Streamlit header/toolbar was always white,
           regardless of theme. Give it the same background as the app so
           it blends in instead of showing as a white bar. ---- */
        [data-testid="stHeader"] {
            background-color: %(header_bg)s !important;
            background: %(header_bg)s !important;
        }
        [data-testid="stHeader"]::before {
            background: %(header_bg)s !important;
        }
        [data-testid="stToolbar"] {
            background-color: transparent !important;
        }
        [data-testid="stDecoration"] {
            background-image: none !important;
            background-color: %(header_bg)s !important;
        }

        /* Remove Streamlit's default extra bottom padding that leaves a
           dead empty gap below the last element */
        .block-container {
            padding-top: 0.5rem;
            padding-bottom: 1rem;
            max-width: 1100px;
            position: relative;
            z-index: 1; /* sits above the starfield */
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

        /* ---- FIX 2: text typed into the chat box was invisible in dark
           mode. Setting `color` alone isn't always enough — Streamlit/the
           browser can apply -webkit-text-fill-color, which wins over a
           plain `color` declaration. Force both, on every selector variant
           that might match the underlying textarea, and also fix the
           caret so the cursor itself is visible. ---- */
        /* NOTE: the chat input's actual rendered background stays white
           in both themes (Streamlit/baseweb keeps its own white textarea
           under the hood, ignoring our translucent input_bg here), so the
           text color must be a fixed dark color too — using the
           theme-dependent input_text (white in dark mode) made the typed
           text invisible on that white box. Hardcode dark text here
           instead of following the theme. */
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInputTextArea"],
        [data-testid="stChatInput"] textarea:focus,
        [data-testid="stChatInput"] textarea:not(:placeholder-shown) {
            background: #FFFFFF !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            caret-color: #111827 !important;
            border-radius: 12px !important;
            border: 1px solid var(--border-color) !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #6B7280 !important;
            -webkit-text-fill-color: #6B7280 !important;
            opacity: 1 !important;
        }

        /* ---- Generic text readability, app-wide ----
           Streamlit's own widgets (captions, labels, markdown) don't know
           about our custom background, so in dark mode we explicitly make
           all of this light-colored. In light mode this just restores the
           original dark-on-white text. */
        .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stExpander"] summary span,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: var(--text-main);
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
            color: %(header_title)s;
        }
        .brand-subtitle {
            color: var(--text-muted);
            font-size: 0.92rem;
        }

        /* ---- Theme toggle, top of every page ---- */
        .theme-toggle-row {
            display: flex;
            justify-content: flex-end;
            margin-bottom: -0.5rem;
        }
        .theme-toggle-row [data-testid="stWidgetLabel"] p {
            color: var(--text-main) !important;
        }

        /* Auth card — smaller, tighter, centered */
        .auth-card {
            max-width: 340px;
            margin: 1.5rem auto 0 auto;
            padding: 1.5rem 1.5rem 1.25rem 1.5rem;
            background: %(auth_card_bg)s;
            border-radius: 14px;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            position: relative;
            z-index: 1; /* sits above the animated stars */
        }
        .auth-title {
            font-weight: 700;
            font-size: 1.15rem;
            color: var(--text-main);
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
            background: %(input_bg)s !important;
            color: %(input_text)s !important;
            -webkit-text-fill-color: %(input_text)s !important;
            border-radius: 8px !important;
            padding: 0.45rem 0.7rem !important;
            font-size: 0.9rem !important;
            border: 1px solid var(--border-color) !important;
        }
        .auth-card [data-testid="stTextInput"] input::placeholder {
            color: %(placeholder)s !important;
            opacity: 1 !important;
        }
        .auth-card [data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
        }
        .auth-card [data-testid="stExpander"] {
            background: var(--panel-alt-bg);
        }

        div.stButton > button, div.stFormSubmitButton > button {
            border-radius: 8px;
            font-weight: 600;
            border: none;
            background-color: var(--brand-primary);
            color: white !important;
            transition: background-color 0.15s ease;
        }
        div.stButton > button:hover, div.stFormSubmitButton > button:hover {
            background-color: var(--brand-primary-dark);
            color: white !important;
        }
        div.stButton > button p, div.stFormSubmitButton > button p {
            color: white !important;
        }

        /* Sidebar polish */
        [data-testid="stSidebar"] {
            background-color: %(sidebar_bg)s;
            border-right: 1px solid var(--border-color);
        }
        [data-testid="stSidebar"] > div {
            position: relative;
            z-index: 1; /* sits above the starfield */
        }

        /* Chat message bubbles */
        [data-testid="stChatMessage"] {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
            position: relative;
            z-index: 1;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span {
            color: var(--text-main) !important;
        }

        .source-card {
            padding: 0.85rem 1rem;
            background: var(--panel-alt-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            margin-bottom: 0.6rem;
        }
        .source-card strong {
            color: var(--text-main) !important;
        }
        .source-card div {
            color: var(--text-muted) !important;
        }
        .source-score {
            display: inline-block;
            padding: 0.05rem 0.5rem;
            border-radius: 6px;
            background: %(badge_bg)s;
            color: %(badge_text)s;
            font-size: 0.78rem;
            font-weight: 600;
        }

        /* Expander polish */
        [data-testid="stExpander"] {
            border: 1px solid var(--border-color);
            border-radius: 10px;
            background: var(--panel-bg);
            position: relative;
            z-index: 1;
        }

        [data-testid="stAlert"] {
            position: relative;
            z-index: 1;
        }

        /* =====================================================
           Animated star background — used app-wide whenever dark mode is
           active (login page, reset page, AND the authenticated app:
           sidebar, chat, everything). It's a fixed, full-screen layer
           behind all content (z-index: 0), so every other panel above
           simply needs z-index: 1 to sit on top of it, which is already
           handled by the rules above.

           Two independent pieces:
             1) `.stars-container` — the original full-width layers
                (.stars/.stars2/.stars3/.stars4), spread across the whole
                screen.
             2) `.stars-right-container` — an EXTRA layer anchored to the
                right edge of the viewport (`right: 0`, fixed width), so
                there's always a denser, visible cluster of stars on the
                right no matter how wide the screen is.
           ===================================================== */
        .stars-container {
            position: fixed;
            inset: 0;
            z-index: 0;
            overflow: hidden;
            background: linear-gradient(180deg, #0b1026 0%%, #141a35 100%%);
            opacity: %(stars_opacity)s;
            pointer-events: none;
        }

        .stars-right-container {
            position: fixed;
            top: 0;
            right: 0;
            width: 420px;
            height: 100%%;
            z-index: 0;
            overflow: hidden;
            opacity: %(stars_opacity)s;
            pointer-events: none;
        }

        .stars, .stars2, .stars3, .stars4,
        .stars-r1, .stars-r2, .stars-r3 {
            position: absolute;
            top: 0;
            left: 0;
            width: 1px;
            height: 1px;
            background: transparent;
        }

        .stars {
            box-shadow:
                173px 924px #fff, 811px 12px #fff, 55px 431px #fff, 909px 271px #fff,
                390px 611px #fff, 22px 780px #fff, 640px 90px #fff, 300px 500px #fff,
                750px 340px #fff, 120px 150px #fff, 480px 820px #fff, 900px 600px #fff,
                60px 60px #fff, 250px 950px #fff, 700px 700px #fff, 15px 300px #fff,
                850px 450px #fff, 400px 100px #fff, 550px 900px #fff, 100px 500px #fff,
                980px 820px #fff, 40px 90px #fff, 620px 430px #fff, 260px 60px #fff,
                760px 980px #fff, 890px 100px #fff, 200px 700px #fff, 330px 330px #fff,
                45px 620px #fff, 510px 40px #fff, 670px 560px #fff, 940px 900px #fff,
                160px 260px #fff, 820px 780px #fff, 380px 870px #fff, 720px 210px #fff;
            animation: animStar 60s linear infinite;
        }
        .stars:after {
            content: " ";
            position: absolute;
            top: 1000px;
            width: 1px;
            height: 1px;
            background: transparent;
            box-shadow: inherit;
        }

        .stars2 {
            box-shadow:
                200px 100px #fff, 500px 300px #fff, 800px 700px #fff, 100px 900px #fff,
                650px 50px #fff, 350px 650px #fff, 900px 400px #fff, 50px 250px #fff,
                750px 850px #fff, 450px 450px #fff, 970px 220px #fff, 20px 620px #fff,
                280px 780px #fff, 610px 920px #fff, 860px 60px #fff, 130px 400px #fff;
            width: 2px;
            height: 2px;
            animation: animStar 100s linear infinite;
        }
        .stars2:after {
            content: " ";
            position: absolute;
            top: 1000px;
            width: 2px;
            height: 2px;
            background: transparent;
            box-shadow: inherit;
        }

        .stars3 {
            box-shadow:
                300px 200px #fff, 700px 600px #fff, 100px 800px #fff, 850px 150px #fff,
                500px 900px #fff, 950px 500px #fff, 60px 480px #fff, 420px 40px #fff,
                220px 920px #fff, 780px 320px #fff;
            width: 3px;
            height: 3px;
            animation: animStar 150s linear infinite;
        }
        .stars3:after {
            content: " ";
            position: absolute;
            top: 1000px;
            width: 3px;
            height: 3px;
            background: transparent;
            box-shadow: inherit;
        }

        /* Extra sparse, larger, slow-drifting layer for a bit more depth */
        .stars4 {
            box-shadow:
                90px 150px #fff, 430px 730px #fff, 610px 210px #fff, 860px 900px #fff,
                260px 500px #fff, 940px 60px #fff;
            width: 2px;
            height: 2px;
            opacity: 0.7;
            animation: animStar 200s linear infinite;
        }
        .stars4:after {
            content: " ";
            position: absolute;
            top: 1000px;
            width: 2px;
            height: 2px;
            background: transparent;
            box-shadow: inherit;
        }

        /* ---- FIX 3: extra stars concentrated on the right side.
           These live inside `.stars-right-container` (fixed width 420px,
           anchored to the right edge), so their x-coordinates only need
           to span 0-420px to always sit near the right of the screen,
           on any monitor size. Three layers, different sizes/speeds for
           parallax depth, denser than the main layers. ---- */
        .stars-r1 {
            box-shadow:
                20px 80px #fff, 60px 240px #fff, 10px 400px #fff, 90px 60px #fff,
                140px 320px #fff, 30px 560px #fff, 180px 140px #fff, 70px 700px #fff,
                220px 480px #fff, 15px 860px #fff, 260px 200px #fff, 110px 920px #fff,
                300px 40px #fff, 45px 640px #fff, 340px 780px #fff, 190px 900px #fff,
                380px 100px #fff, 250px 620px #fff, 400px 340px #fff, 130px 460px #fff;
            animation: animStar 55s linear infinite;
        }
        .stars-r1:after {
            content: " ";
            position: absolute;
            top: 1000px;
            width: 1px;
            height: 1px;
            background: transparent;
            box-shadow: inherit;
        }

        .stars-r2 {
            box-shadow:
                50px 120px #fff, 150px 500px #fff, 250px 800px #fff, 350px 200px #fff,
                80px 640px #fff, 200px 60px #fff, 320px 460px #fff, 20px 300px #fff,
                290px 920px #fff, 120px 900px #fff;
            width: 2px;
            height: 2px;
            animation: animStar 90s linear infinite;
        }
        .stars-r2:after {
            content: " ";
            position: absolute;
            top: 1000px;
            width: 2px;
            height: 2px;
            background: transparent;
            box-shadow: inherit;
        }

        .stars-r3 {
            box-shadow:
                100px 180px #fff, 260px 620px #fff, 40px 860px #fff, 370px 60px #fff,
                180px 940px #fff, 320px 340px #fff;
            width: 3px;
            height: 3px;
            opacity: 0.8;
            animation: animStar 140s linear infinite;
        }
        .stars-r3:after {
            content: " ";
            position: absolute;
            top: 1000px;
            width: 3px;
            height: 3px;
            background: transparent;
            box-shadow: inherit;
        }

        @keyframes animStar {
            from { transform: translateY(0px); }
            to   { transform: translateY(-1000px); } /* stars drift upward */
        }
    </style>
    """ % _theme,
    unsafe_allow_html=True,
)


def render_stars_background():
    """Injects the fixed, full-screen animated starfield, PLUS a dedicated
    right-edge cluster (`.stars-right-container`) for a denser, always-
    visible group of stars on the right side of the screen. Called once
    per run, near the top, whenever dark mode is active — both containers
    sit behind every page (login, reset, and the authenticated app) since
    they're position: fixed with z-index: 0, and every panel on top
    already has z-index: 1. In light mode both render with opacity 0, so
    they're effectively invisible and the app looks exactly like the
    original plain light theme."""
    st.markdown(
        """
        <div class="stars-container">
            <div class="stars"></div>
            <div class="stars2"></div>
            <div class="stars3"></div>
            <div class="stars4"></div>
        </div>
        <div class="stars-right-container">
            <div class="stars-r1"></div>
            <div class="stars-r2"></div>
            <div class="stars-r3"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_theme_toggle():
    """Small light/dark toggle shown at the top of every page (login,
    reset-password, and the main authenticated app). Flipping it just
    updates st.session_state.theme_mode and reruns — all the styling
    above reacts automatically since it's built from THEMES[theme_mode]."""
    st.markdown('<div class="theme-toggle-row">', unsafe_allow_html=True)
    is_dark = st.session_state.theme_mode == "dark"
    new_is_dark = st.toggle(
        "🌙 Dark" if is_dark else "☀️ Light",
        value=is_dark,
        key="theme_mode_toggle_widget",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    if new_is_dark != is_dark:
        st.session_state.theme_mode = "dark" if new_is_dark else "light"
        st.rerun()


# Starfield sits behind absolutely everything on the page, on every run.
render_stars_background()
# Toggle is the very first interactive element on every page.
render_theme_toggle()


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
        <div class="brand-header" style="justify-content:center; margin-top:1rem; position:relative; z-index:1;">
            <span style="font-size:1.8rem;">📚</span>
            <span class="brand-title">ML Research Assistant</span>
        </div>
        <div class="brand-subtitle" style="text-align:center; margin-bottom: 0.5rem; position:relative; z-index:1;">
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
        <div class="brand-header" style="justify-content:center; margin-top:1rem; position:relative; z-index:1;">
            <span style="font-size:1.8rem;">📚</span>
            <span class="brand-title">ML Research Assistant</span>
        </div>
        <div class="brand-subtitle" style="text-align:center; margin-bottom: 0.5rem; position:relative; z-index:1;">
            Choose a new password
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    # If we already successfully reset the password for this token in a
    # previous rerun, don't re-verify the token — it's now marked "used"
    # in the database (by design, so it can't be replayed), so re-checking
    # it here would incorrectly show "invalid or expired" the moment the
    # user interacts with anything on this success screen (e.g. clicking
    # "Go to sign in" itself triggers a rerun, which would hit that check).
    if st.session_state.get("reset_done_token") == token:
        st.success("Your password has been reset. You can now sign in with your new password.")
        if st.button("Go to sign in", use_container_width=True):
            st.session_state.pop("reset_done_token", None)
            st.query_params.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

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
                    # Mark success in session_state and rerun immediately —
                    # the next run of this function takes the branch above,
                    # showing a clean success screen without re-verifying
                    # the now-used token.
                    st.session_state["reset_done_token"] = token
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

            col_load, col_menu = st.columns([5, 1])
            with col_load:
                if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                    st.session_state.conversation_id = conv["id"]
                    st.session_state.messages = history.load_messages(conv["id"])
                    st.rerun()
            with col_menu:
                with st.popover("⋮", use_container_width=True):
                    new_title = st.text_input(
                        "Rename conversation",
                        value=conv["title"],
                        key=f"rename_input_{conv['id']}",
                    )
                    col_rename, col_delete = st.columns(2)
                    with col_rename:
                        if st.button("Rename", key=f"rename_btn_{conv['id']}", use_container_width=True):
                            if history.rename_conversation(conv["id"], new_title):
                                st.rerun()
                            else:
                                st.error("Title can't be empty.")
                    with col_delete:
                        if st.button("Delete", key=f"del_{conv['id']}", use_container_width=True):
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
                    <div style="font-size:0.85rem; margin-top:0.4rem;">
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