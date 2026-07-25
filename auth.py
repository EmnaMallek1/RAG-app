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

Requires TURSO_DATABASE_URL and TURSO_AUTH_TOKEN to be set in
`.streamlit/secrets.toml` (locally) or the Streamlit Cloud Secrets panel
(deployed).
"""

import hashlib
import os
import re
import time

import streamlit as st
from turso_python.connection import TursoConnection

PBKDF2_ITERATIONS = 200_000

_TABLE_READY = False  # ensures CREATE TABLE only runs once per session


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