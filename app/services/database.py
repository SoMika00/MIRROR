"""
SQLite database service for MIRROR.

Tables:
  - users: anonymous user sessions (cookie-based)
  - conversations: chat conversation threads per user
  - messages: individual messages within conversations
  - user_sources: tracks which sources each user has uploaded/scraped
  - logs: structured application logs
"""

import sqlite3
import uuid
import time
import json
import logging
import threading
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = "./data/mirror.db"


class Database:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    @contextmanager
    def get_cursor(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def init_db(self):
        """Create all tables if they don't exist."""
        import os
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        with self.get_cursor() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    display_name TEXT DEFAULT 'Anonymous'
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT DEFAULT 'New conversation',
                    mode TEXT NOT NULL DEFAULT 'chat',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    mode TEXT DEFAULT 'chat',
                    sources TEXT,
                    timings TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS user_sources (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_type TEXT NOT NULL CHECK(source_type IN ('document', 'web', 'article')),
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    user_id TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_user_sources_user ON user_sources(user_id);
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp DESC);
            """)
        logger.info("Database initialized")

    # --- Users ---
    def get_or_create_user(self, user_id: Optional[str] = None) -> str:
        now = time.time()
        if user_id:
            with self.get_cursor() as c:
                c.execute("SELECT id FROM users WHERE id = ?", (user_id,))
                if c.fetchone():
                    c.execute("UPDATE users SET last_seen = ? WHERE id = ?", (now, user_id))
                    return user_id
        # Create new user
        new_id = str(uuid.uuid4())
        with self.get_cursor() as c:
            c.execute("INSERT INTO users (id, created_at, last_seen) VALUES (?, ?, ?)",
                       (new_id, now, now))
        return new_id

    # --- Conversations ---
    def create_conversation(self, user_id: str, mode: str = "chat", title: str = "New conversation") -> str:
        conv_id = str(uuid.uuid4())
        now = time.time()
        with self.get_cursor() as c:
            c.execute("INSERT INTO conversations (id, user_id, title, mode, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                       (conv_id, user_id, title, mode, now, now))
        return conv_id

    def get_conversations(self, user_id: str, limit: int = 50) -> List[Dict]:
        with self.get_cursor() as c:
            c.execute("""
                SELECT c.id, c.title, c.mode, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as msg_count
                FROM conversations c
                WHERE c.user_id = ?
                ORDER BY c.updated_at DESC
                LIMIT ?
            """, (user_id, limit))
            return [dict(row) for row in c.fetchall()]

    def delete_conversation(self, conv_id: str, user_id: str) -> bool:
        with self.get_cursor() as c:
            c.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
            return c.rowcount > 0

    def user_owns_conversation(self, conv_id: str, user_id: str) -> bool:
        """Check if a conversation belongs to the given user."""
        with self.get_cursor() as c:
            c.execute("SELECT 1 FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id))
            return c.fetchone() is not None

    def update_conversation_title(self, conv_id: str, title: str):
        with self.get_cursor() as c:
            c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))

    # --- Messages ---
    def add_message(self, conversation_id: str, role: str, content: str,
                    mode: str = "chat", sources: Optional[List] = None,
                    timings: Optional[Dict] = None) -> str:
        msg_id = str(uuid.uuid4())
        now = time.time()
        with self.get_cursor() as c:
            c.execute("""
                INSERT INTO messages (id, conversation_id, role, content, mode, sources, timings, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (msg_id, conversation_id, role, content, mode,
                  json.dumps(sources) if sources else None,
                  json.dumps(timings) if timings else None,
                  now))
            c.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
        return msg_id

    def get_messages(self, conversation_id: str, limit: int = 100) -> List[Dict]:
        with self.get_cursor() as c:
            c.execute("""
                SELECT id, role, content, mode, sources, timings, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                LIMIT ?
            """, (conversation_id, limit))
            rows = []
            for row in c.fetchall():
                d = dict(row)
                if d["sources"]:
                    d["sources"] = json.loads(d["sources"])
                if d["timings"]:
                    d["timings"] = json.loads(d["timings"])
                rows.append(d)
            return rows

    def get_recent_context(self, conversation_id: str, n: int = 6) -> List[Dict]:
        """Get last n messages for conversation context."""
        with self.get_cursor() as c:
            c.execute("""
                SELECT role, content FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (conversation_id, n))
            rows = [dict(r) for r in c.fetchall()]
            rows.reverse()
            return rows

    # --- User Sources ---
    def add_user_source(self, user_id: str, source_name: str, source_type: str,
                        metadata: Optional[Dict] = None) -> str:
        src_id = str(uuid.uuid4())
        now = time.time()
        with self.get_cursor() as c:
            c.execute("""
                INSERT INTO user_sources (id, user_id, source_name, source_type, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (src_id, user_id, source_name, source_type,
                  json.dumps(metadata) if metadata else None, now))
        return src_id

    def get_user_sources(self, user_id: str) -> List[Dict]:
        with self.get_cursor() as c:
            c.execute("""
                SELECT id, source_name, source_type, metadata, created_at
                FROM user_sources
                WHERE user_id = ?
                ORDER BY created_at DESC
            """, (user_id,))
            rows = []
            for row in c.fetchall():
                d = dict(row)
                if d["metadata"]:
                    d["metadata"] = json.loads(d["metadata"])
                rows.append(d)
            return rows

    def delete_user_source(self, source_id: str, user_id: str) -> Optional[str]:
        """Delete a user source, returns source_name for Qdrant cleanup."""
        with self.get_cursor() as c:
            c.execute("SELECT source_name FROM user_sources WHERE id = ? AND user_id = ?",
                       (source_id, user_id))
            row = c.fetchone()
            if row:
                c.execute("DELETE FROM user_sources WHERE id = ? AND user_id = ?", (source_id, user_id))
                return row["source_name"]
        return None

    # --- Logs ---
    def add_log(self, level: str, component: str, message: str,
                details: Optional[Dict] = None, user_id: Optional[str] = None):
        with self.get_cursor() as c:
            c.execute("""
                INSERT INTO logs (timestamp, level, component, message, details, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (time.time(), level, component, message,
                  json.dumps(details) if details else None, user_id))

    def get_logs(self, limit: int = 100, component: Optional[str] = None,
                 level: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM logs"
        params = []
        conditions = []
        if component:
            conditions.append("component = ?")
            params.append(component)
        if level:
            conditions.append("level = ?")
            params.append(level)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self.get_cursor() as c:
            c.execute(query, params)
            rows = []
            for row in c.fetchall():
                d = dict(row)
                if d.get("details"):
                    d["details"] = json.loads(d["details"])
                rows.append(d)
            return rows


db = Database()
