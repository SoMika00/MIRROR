"""Web scraping routes - scrape a URL and ask questions about it."""

import time
from flask import Blueprint, request, jsonify

from app.services.scraper import scrape_url
from app.services.pdf_parser import chunk_text
from app.services.embedding import embedding_service
from app.services.qdrant_store import qdrant_store
from app.services.database import db
from app.config import rag_cfg

scraper_bp = Blueprint("scraper", __name__)

# In-memory cache of scraped pages for follow-up questions (bounded to 20 entries)
_MAX_CACHE = 20
_scraped_cache = {}


def _get_user_id() -> str:
    """Get or create user from cookie."""
    user_id = request.cookies.get("mirror_uid")
    return db.get_or_create_user(user_id)


@scraper_bp.route("/scrape", methods=["POST"])
def scrape():
    """Scrape a URL and return its content. Optionally index it."""
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field"}), 400

    url = data["url"].strip()
    index = data.get("index", False)
    user_id = _get_user_id()

    try:
        result = scrape_url(url)

        # Cache for follow-up questions (evict oldest if over limit)
        if len(_scraped_cache) >= _MAX_CACHE and url not in _scraped_cache:
            oldest = next(iter(_scraped_cache))
            del _scraped_cache[oldest]
        _scraped_cache[url] = result

        # Optionally index into vector store
        if index:
            chunks = chunk_text(result["text"], rag_cfg.chunk_size, rag_cfg.chunk_overlap)
            if chunks:
                vectors = embedding_service.encode(chunks).tolist()
                payloads = [
                    {
                        "source_name": url,
                        "source_type": "web",
                        "page": 1,
                        "chunk_index": i,
                        "url": url,
                        "title": result.get("title", ""),
                        "user_id": user_id,
                    }
                    for i in range(len(chunks))
                ]
                qdrant_store.upsert(texts=chunks, vectors=vectors, payloads=payloads)
                result["indexed"] = True
                result["chunks_indexed"] = len(chunks)

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@scraper_bp.route("/ask", methods=["POST"])
def ask_about_page():
    """Ask a question about a previously scraped page."""
    data = request.get_json()
    if not data or "url" not in data or "question" not in data:
        return jsonify({"error": "Missing 'url' or 'question'"}), 400

    url = data["url"].strip()
    question = data["question"].strip()

    cached = _scraped_cache.get(url)
    if not cached:
        return jsonify({"error": "URL not scraped yet. Scrape it first."}), 404

    from app.services.rag import query_scraped_content
    from app.services.llm import llm_service

    if not llm_service.is_loaded():
        return jsonify({"error": "No LLM loaded."}), 503

    try:
        result = query_scraped_content(
            question=question,
            url=url,
            title=cached.get("title", ""),
            content=cached["text"],
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@scraper_bp.route("/cached", methods=["GET"])
def list_cached():
    """List currently cached scraped pages."""
    return jsonify({
        "pages": [
            {"url": url, "title": data.get("title", ""), "chars": data.get("char_count", 0)}
            for url, data in _scraped_cache.items()
        ]
    })
