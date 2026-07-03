"""
Hybrid retrieval service - zero local models, zero external infrastructure.

Design:
  - Lexical search: SQLite FTS5 (BM25) - in-process, no RAM overhead, no service
    to operate. On a portfolio-scale corpus (hundreds of chunks) BM25 keyword
    retrieval is competitive with dense retrieval and costs nothing.
  - Semantic search (optional): embeddings fetched from an OpenAI-compatible
    API (set EMBEDDINGS_API_KEY). Vectors are stored as BLOBs next to the text
    and scored in-process with cosine similarity. When enabled, scores are
    fused: hybrid = w * vector + (1 - w) * bm25.

This replaced a self-hosted stack (Qdrant + BGE-M3 + CrossEncoder, ~8 GB RAM)
when the site moved to an API-first architecture on a small VPS.
"""

import array
import logging
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import rag_cfg, embeddings_api_cfg

logger = logging.getLogger(__name__)

DB_PATH = "./data/mirror.db"


@dataclass
class SearchResult:
    text: str
    score: float
    source: str
    source_type: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


def _vector_to_blob(vec: List[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _blob_to_vector(blob: bytes) -> List[float]:
    a = array.array("f")
    a.frombytes(blob)
    return list(a)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _fts_query(question: str) -> str:
    """Sanitize free text into a safe FTS5 OR-query."""
    tokens = re.findall(r"\w+", question.lower(), flags=re.UNICODE)
    tokens = [t for t in tokens if len(t) > 1][:24]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


class RetrievalStore:
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

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def init_db(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                page INTEGER,
                chunk_index INTEGER,
                user_id TEXT,
                text TEXT NOT NULL,
                embedding BLOB,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_name, user_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
        """)
        conn.commit()
        logger.info("Retrieval store initialized (SQLite FTS5)")

    # --- Indexing ---

    def upsert(self, texts: List[str], payloads: List[Dict[str, Any]],
               vectors: Optional[List[List[float]]] = None):
        """Insert chunks. Embeds via API when the embeddings client is enabled."""
        if vectors is None and embeddings_api_cfg.enabled:
            from app.services.embeddings_api import embeddings_client
            try:
                vectors = embeddings_client.embed(texts)
            except Exception as e:
                logger.warning(f"Embeddings API failed, indexing lexical-only: {e}")
                vectors = None

        conn = self._conn()
        now = time.time()
        rows = []
        for i, (text, payload) in enumerate(zip(texts, payloads)):
            blob = _vector_to_blob(vectors[i]) if vectors else None
            rows.append((
                payload.get("source_name", "unknown"),
                payload.get("source_type", "unknown"),
                payload.get("page"),
                payload.get("chunk_index", i),
                payload.get("user_id"),
                text,
                blob,
                now,
            ))
        conn.executemany(
            "INSERT INTO chunks (source_name, source_type, page, chunk_index, user_id, text, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        logger.info(f"Indexed {len(rows)} chunks (vectors: {'yes' if vectors else 'no'})")

    def delete_by_source(self, source_name: str, user_id: Optional[str] = None):
        conn = self._conn()
        if user_id:
            conn.execute("DELETE FROM chunks WHERE source_name = ? AND user_id = ?", (source_name, user_id))
        else:
            conn.execute("DELETE FROM chunks WHERE source_name = ?", (source_name,))
        conn.commit()
        logger.info(f"Deleted chunks for source: {source_name}")

    # --- Search ---

    def search(self, question: str, top_k: int = 8,
               source_type: Optional[str] = None,
               source_names: Optional[List[str]] = None,
               user_id: Optional[str] = None,
               include_global: bool = True) -> List[SearchResult]:
        """Hybrid search: BM25 always; cosine fusion when embeddings enabled.

        Visibility: global chunks (user_id IS NULL, i.e. portfolio content)
        plus the requesting user's own uploads.
        """
        match = _fts_query(question)
        if not match:
            return []

        conditions = ["chunks_fts MATCH ?"]
        params: List[Any] = [match]

        if include_global and user_id:
            conditions.append("(c.user_id IS NULL OR c.user_id = ?)")
            params.append(user_id)
        elif user_id:
            conditions.append("c.user_id = ?")
            params.append(user_id)
        else:
            conditions.append("c.user_id IS NULL")

        if source_type:
            conditions.append("c.source_type = ?")
            params.append(source_type)
        if source_names:
            placeholders = ",".join("?" for _ in source_names)
            conditions.append(f"(c.source_name IN ({placeholders}) OR c.user_id IS NULL)")
            params.extend(source_names)

        # Fetch a wider candidate pool for hybrid re-scoring
        pool = top_k * 4 if embeddings_api_cfg.enabled else top_k
        params.append(pool)

        start = time.time()
        rows = self._conn().execute(f"""
            SELECT c.id, c.source_name, c.source_type, c.page, c.chunk_index,
                   c.text, c.embedding, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE {' AND '.join(conditions)}
            ORDER BY rank
            LIMIT ?
        """, params).fetchall()

        # bm25() is lower-is-better; map to (0, 1]
        results = []
        for r in rows:
            bm25_score = 1.0 / (1.0 + max(r["rank"], 0.0))
            results.append((r, bm25_score))

        # Optional dense re-scoring via embeddings API
        if embeddings_api_cfg.enabled and results:
            try:
                from app.services.embeddings_api import embeddings_client
                qvec = embeddings_client.embed([question])[0]
                w = rag_cfg.hybrid_vector_weight
                rescored = []
                for r, bm25_score in results:
                    if r["embedding"]:
                        cos = _cosine(qvec, _blob_to_vector(r["embedding"]))
                        score = w * cos + (1 - w) * bm25_score
                    else:
                        score = bm25_score
                    rescored.append((r, score))
                results = rescored
            except Exception as e:
                logger.warning(f"Hybrid rescoring skipped: {e}")

        results.sort(key=lambda x: x[1], reverse=True)
        elapsed = (time.time() - start) * 1000
        logger.debug(f"Search '{question[:40]}' -> {len(results)} results in {elapsed:.1f}ms")

        return [
            SearchResult(
                text=r["text"],
                score=round(s, 4),
                source=r["source_name"],
                source_type=r["source_type"],
                page=r["page"],
                chunk_index=r["chunk_index"],
            )
            for r, s in results[:top_k]
        ]

    # --- Introspection ---

    def list_sources(self, user_id: Optional[str] = None,
                     include_global: bool = False) -> List[Dict[str, Any]]:
        conn = self._conn()
        if user_id and include_global:
            rows = conn.execute("""
                SELECT source_name, source_type, COUNT(*) AS chunks
                FROM chunks WHERE user_id = ? OR user_id IS NULL
                GROUP BY source_name, source_type
            """, (user_id,)).fetchall()
        elif user_id:
            rows = conn.execute("""
                SELECT source_name, source_type, COUNT(*) AS chunks
                FROM chunks WHERE user_id = ?
                GROUP BY source_name, source_type
            """, (user_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT source_name, source_type, COUNT(*) AS chunks
                FROM chunks WHERE user_id IS NULL
                GROUP BY source_name, source_type
            """).fetchall()
        return [dict(r) for r in rows]

    def get_info(self) -> dict:
        conn = self._conn()
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        with_vec = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
        return {
            "engine": "SQLite FTS5 (BM25)" + (" + API embeddings" if embeddings_api_cfg.enabled else ""),
            "connected": True,
            "points_count": total,
            "chunks_with_vectors": with_vec,
            "hybrid_enabled": embeddings_api_cfg.enabled,
        }


retrieval_store = RetrievalStore()
