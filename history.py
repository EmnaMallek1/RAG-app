"""
history.py
----------
Persists chat conversations per user in Turso, so users see their past
conversations again after logging back in.

Two tables:
- conversations(id, user_email, title, created_at)
- chat_messages(id, conversation_id, role, content, sources, created_at)

Uses the same low-level approach as auth.py: talks to TursoConnection
directly with plain SQL rather than the buggy TursoCRUD/TursoSchemaManager
helpers, and stores all timestamps as TEXT (not REAL/float) to avoid a
known bug in turso-python's float argument serialization.
"""

import json
import time
import uuid

import streamlit as st
from turso_python.connection import TursoConnection

_TABLES_READY = False


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


def _ensure_tables():
    global _TABLES_READY
    if _TABLES_READY:
        return

    conn = _get_connection()
    conn.execute_query(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_email TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute_query(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    _TABLES_READY = True


def create_conversation(user_email: str, title: str) -> str:
    """Creates a new conversation for this user and returns its id."""
    _ensure_tables()
    conn = _get_connection()

    conversation_id = str(uuid.uuid4())
    title = (title or "New conversation").strip()[:60]

    conn.execute_query(
        "INSERT INTO conversations (id, user_email, title, created_at) VALUES (?, ?, ?, ?)",
        [conversation_id, user_email.strip().lower(), title, str(time.time())],
    )
    return conversation_id


def save_message(conversation_id: str, role: str, content: str, sources=None):
    """Appends a single message to a conversation."""
    _ensure_tables()
    conn = _get_connection()

    sources_json = json.dumps(sources) if sources else None

    conn.execute_query(
        "INSERT INTO chat_messages (id, conversation_id, role, content, sources, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [str(uuid.uuid4()), conversation_id, role, content, sources_json, str(time.time())],
    )


def list_conversations(user_email: str) -> list[dict]:
    """Returns this user's conversations, most recent first."""
    _ensure_tables()
    conn = _get_connection()

    rows = _parse_rows(
        conn.execute_query(
            "SELECT id, title, created_at FROM conversations "
            "WHERE user_email = ? ORDER BY created_at DESC",
            [user_email.strip().lower()],
        )
    )
    return rows


def load_messages(conversation_id: str) -> list[dict]:
    """Returns all messages in a conversation, in order, with sources
    parsed back into Python objects."""
    _ensure_tables()
    conn = _get_connection()

    rows = _parse_rows(
        conn.execute_query(
            "SELECT role, content, sources FROM chat_messages "
            "WHERE conversation_id = ? ORDER BY created_at ASC",
            [conversation_id],
        )
    )

    messages = []
    for row in rows:
        sources = json.loads(row["sources"]) if row.get("sources") else []
        messages.append({
            "role": row["role"],
            "content": row["content"],
            "sources": sources,
        })
    return messages


def delete_conversation(conversation_id: str):
    """Deletes a conversation and all of its messages."""
    _ensure_tables()
    conn = _get_connection()
    conn.execute_query("DELETE FROM chat_messages WHERE conversation_id = ?", [conversation_id])
    conn.execute_query("DELETE FROM conversations WHERE id = ?", [conversation_id])