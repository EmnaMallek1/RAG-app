"""
auth.py
-------
Email/password authentication for the Streamlit app, backed by a Turso
(cloud-hosted SQLite-compatible) database instead of a local file.

- Users are stored remotely in Turso, so accounts persist across app
  restarts/redeploys on free hosting platforms.
- Passwords are never stored in plain text: each password is hashed with
  PBKDF2-HMAC-SHA256 and a random per-user salt (200,000 iterations).
- Uses `turso_python.connection.TursoConnection` directly with plain SQL
  (rather than the package's TursoCRUD/TursoSchemaManager helpers, which
  have bugs: TursoSchemaManager expects a raw URL instead of a connection
  object, and TursoCRUD.read() returns unparsed raw JSON). Talking to
  TursoConnection.execute_query() directly and parsing the response
  ourselves avoids both issues.
- Password reset: a one-time, time-limited token is generated and stored
  in a `password_resets` table, then emailed to the user as a link via
  Gmail SMTP. Clicking the link lets them set a new password.

Requires TURSO_DATABASE_URL and TURSO_AUTH_TOKEN, and (for password reset)
GMAIL_ADDRESS, GMAIL_APP_PASSWORD, and APP_URL, all set in
`.streamlit/secrets.toml` (locally) or the Streamlit Cloud Secrets panel
(deployed).
"""

import hashlib
import os
import re
import secrets
import smtplib
import time
from email.mime.text import MIMEText

import streamlit as st
from turso_python.connection import TursoConnection

PBKDF2_ITERATIONS = 200_000
RESET_TOKEN_TTL_SECONDS = 30 * 60  # reset links are valid for 30 minutes

_TABLE_READY = False        # ensures the users table CREATE only runs once per session
_RESET_TABLE_READY = False  # ensures the password_resets table CREATE only runs once per session


def _get_connection():
    return TursoConnection(
        database_url=st.secrets["TURSO_DATABASE_URL"],
        auth_token=st.secrets["TURSO_AUTH_TOKEN"],
    )


def _parse_rows(result: dict) -> list[dict]:
    """Turns Turso's raw pipeline JSON response into a list of plain
    {column_name: value} dicts for the first executed statement."""
    try:
        stmt_result = result["results"][0]["response"]["result"]
    except (KeyError, IndexError, TypeError):
        return []

    cols = [c["name"] for c in stmt_result.get("cols", [])]
    rows = stmt_result.get("rows", [])

    parsed = []
    for row in rows:
        record = {}
        for col_name, cell in zip(cols, row):
            record[col_name] = None if cell.get("type") == "null" else cell.get("value")
        parsed.append(record)
    return parsed


def _ensure_table():
    global _TABLE_READY
    if _TABLE_READY:
        return

    conn = _get_connection()
    conn.execute_query(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    _TABLE_READY = True


def _ensure_reset_table():
    global _RESET_TABLE_READY
    if _RESET_TABLE_READY:
        return

    conn = _get_connection()
    conn.execute_query(
        """
        CREATE TABLE IF NOT EXISTS password_resets (
            token TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    _RESET_TABLE_READY = True


def _hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    ).hex()


def _is_valid_email(email: str) -> bool:
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) is not None


def create_user(email: str, name: str, password: str):
    """Register a new user. Returns (success: bool, message: str)."""
    email = email.strip().lower()
    name = name.strip()

    if not _is_valid_email(email):
        return False, "Please enter a valid email address."
    if not name:
        return False, "Please enter your name."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."

    _ensure_table()
    conn = _get_connection()

    existing = _parse_rows(
        conn.execute_query("SELECT email FROM users WHERE email = ?", [email])
    )
    if existing:
        return False, "An account with this email already exists."

    salt = os.urandom(16).hex()
    password_hash = _hash_password(password, salt)

    conn.execute_query(
        "INSERT INTO users (email, name, salt, password_hash, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [email, name, salt, password_hash, str(time.time())],
    )

    return True, "Account created successfully. You can now sign in."


def verify_user(email: str, password: str):
    """Check credentials. Returns (success: bool, name: str | None, message: str)."""
    email = email.strip().lower()

    _ensure_table()
    conn = _get_connection()

    rows = _parse_rows(
        conn.execute_query(
            "SELECT name, salt, password_hash FROM users WHERE email = ?", [email]
        )
    )
    if not rows:
        return False, None, "No account found with this email."

    row = rows[0]
    if _hash_password(password, row["salt"]) != row["password_hash"]:
        return False, None, "Incorrect password."

    return True, row["name"], "Login successful."


def user_exists(email: str) -> bool:
    email = email.strip().lower()
    _ensure_table()
    conn = _get_connection()
    rows = _parse_rows(
        conn.execute_query("SELECT email FROM users WHERE email = ?", [email])
    )
    return len(rows) > 0


# =========================================================
# Password reset
# =========================================================
def _send_reset_email(to_email: str, reset_link: str) -> bool:
    """Sends the reset link via Gmail SMTP. Returns True on success."""
    try:
        gmail_address = st.secrets["GMAIL_ADDRESS"]
        gmail_app_password = st.secrets["GMAIL_APP_PASSWORD"]
    except Exception:
        return False

    subject = "Reset your ML Research Assistant password"
    body = (
        "Hello,\n\n"
        "We received a request to reset your password for the ML Research "
        "Assistant app.\n\n"
        "Click the link below to choose a new password. This link is valid "
        "for 30 minutes and can only be used once:\n\n"
        f"{reset_link}\n\n"
        "If you didn't request this, you can safely ignore this email — "
        "your password will not be changed."
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = to_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(gmail_address, gmail_app_password)
            server.sendmail(gmail_address, [to_email], msg.as_string())
        return True
    except Exception:
        return False


def request_password_reset(email: str, app_url: str):
    """Generates a one-time reset token and emails a reset link.

    Always returns a generic success message even if no account exists
    for that email — this prevents the reset form from being usable to
    check which emails are registered.

    Returns (success: bool, message: str). `success` is only False for
    actual failures (e.g. email couldn't be sent), not for "no such
    account", which is intentionally reported as if it succeeded.
    """
    email = email.strip().lower()
    generic_message = "If an account exists with this email, a reset link has been sent."

    if not _is_valid_email(email):
        return False, "Please enter a valid email address."

    _ensure_table()
    _ensure_reset_table()
    conn = _get_connection()

    rows = _parse_rows(
        conn.execute_query("SELECT email FROM users WHERE email = ?", [email])
    )
    if not rows:
        return True, generic_message  # don't reveal whether the account exists

    token = secrets.token_urlsafe(32)
    expires_at = str(time.time() + RESET_TOKEN_TTL_SECONDS)

    conn.execute_query(
        "INSERT INTO password_resets (token, email, expires_at, used) VALUES (?, ?, ?, 0)",
        [token, email, expires_at],
    )

    separator = "&" if "?" in app_url else "?"
    reset_link = f"{app_url}{separator}reset_token={token}"

    if not _send_reset_email(email, reset_link):
        return False, "Could not send the reset email right now. Please try again later."

    return True, generic_message


def verify_reset_token(token: str):
    """Returns the email tied to a valid, unused, non-expired token, or
    None if the token is invalid/expired/already used."""
    if not token:
        return None

    _ensure_reset_table()
    conn = _get_connection()

    rows = _parse_rows(
        conn.execute_query(
            "SELECT email, expires_at, used FROM password_resets WHERE token = ?",
            [token],
        )
    )
    if not rows:
        return None

    row = rows[0]
    if int(row["used"]) == 1:
        return None
    if float(row["expires_at"]) < time.time():
        return None

    return row["email"]


def reset_password(token: str, new_password: str):
    """Sets a new password for the account tied to `token` and marks the
    token as used so it can't be replayed. Returns (success, message)."""
    if len(new_password) < 8:
        return False, "Password must be at least 8 characters long."

    email = verify_reset_token(token)
    if not email:
        return False, "This reset link is invalid or has expired. Please request a new one."

    _ensure_table()
    _ensure_reset_table()
    conn = _get_connection()

    salt = os.urandom(16).hex()
    password_hash = _hash_password(new_password, salt)

    conn.execute_query(
        "UPDATE users SET salt = ?, password_hash = ? WHERE email = ?",
        [salt, password_hash, email],
    )
    conn.execute_query(
        "UPDATE password_resets SET used = 1 WHERE token = ?", [token]
    )

    return True, "Your password has been reset. You can now sign in with your new password."