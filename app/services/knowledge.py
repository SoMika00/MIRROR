"""
Portfolio knowledge indexer.

At startup, indexes the site's own content (profile/CV, articles, architecture
docs) into the retrieval store as *global* sources (user_id IS NULL), so the
AI chat can answer questions about Michail with citations out of the box -
no upload needed by visitors.

Re-indexing is idempotent: global chunks are wiped and rebuilt on each boot
(the corpus is small; this takes well under a second).
"""

import glob
import logging
import os
import re

from app.services.retrieval import retrieval_store
from app.services.pdf_parser import chunk_text
from app.config import rag_cfg

logger = logging.getLogger(__name__)

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "content", "profile.md")


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text


def _strip_markdown(text: str) -> str:
    """Light cleanup so chunks read naturally in LLM context."""
    text = re.sub(r"```[\s\S]*?```", "", text)          # code blocks
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)   # headings markers
    text = re.sub(r"\*\*?|__?", "", text)                # bold/italic
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links -> label
    return text


def _index_file(filepath: str, source_name: str, source_type: str) -> int:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        logger.warning(f"Cannot read {filepath}: {e}")
        return 0

    body = _strip_markdown(_strip_frontmatter(raw))
    chunks = chunk_text(body, rag_cfg.chunk_size, rag_cfg.chunk_overlap)
    if not chunks:
        return 0

    payloads = [
        {
            "source_name": source_name,
            "source_type": source_type,
            "page": None,
            "chunk_index": i,
            "user_id": None,  # global: visible to every visitor
        }
        for i in range(len(chunks))
    ]
    retrieval_store.upsert(texts=chunks, payloads=payloads)
    return len(chunks)


def index_portfolio_content(articles_dir: str = "./articles", docs_dir: str = "./docs"):
    """Wipe and rebuild all global (portfolio) chunks."""
    conn = retrieval_store._conn()
    conn.execute("DELETE FROM chunks WHERE user_id IS NULL")
    conn.commit()

    total = 0

    # 1. Profile / CV
    if os.path.exists(PROFILE_PATH):
        total += _index_file(PROFILE_PATH, "Profile - Michail Berjaoui", "portfolio")

    # 2. Articles (index French + English variants; the retriever is language-agnostic)
    for fp in sorted(glob.glob(os.path.join(articles_dir, "*.md"))):
        name = os.path.splitext(os.path.basename(fp))[0]
        total += _index_file(fp, f"Article: {name}", "article")

    # 3. Architecture docs
    for fp in sorted(glob.glob(os.path.join(docs_dir, "*.md"))):
        name = os.path.splitext(os.path.basename(fp))[0]
        total += _index_file(fp, f"Docs: {name}", "portfolio")

    logger.info(f"Portfolio knowledge indexed: {total} chunks")
    return total
