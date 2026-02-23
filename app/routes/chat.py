"""Chat/RAG API routes."""

import json
from flask import Blueprint, request, jsonify, Response, stream_with_context

from app.services.rag import query_rag, query_rag_stream, query_scraped_content
from app.services.llm import llm_service

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/query", methods=["POST"])
def chat_query():
    """RAG query with source citations."""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()
    source_type = data.get("source_type")

    if not question:
        return jsonify({"error": "Empty question"}), 400

    if not llm_service.is_loaded():
        return jsonify({"error": "No LLM loaded. Please load a model first via the model manager."}), 503

    try:
        result = query_rag(question, source_type=source_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/stream", methods=["POST"])
def chat_stream():
    """Streaming RAG query."""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()
    source_type = data.get("source_type")

    if not llm_service.is_loaded():
        return jsonify({"error": "No LLM loaded."}), 503

    def generate():
        for token in query_rag_stream(question, source_type=source_type):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@chat_bp.route("/scraper-query", methods=["POST"])
def scraper_query():
    """Query over scraped web content."""
    data = request.get_json()
    if not data or "question" not in data or "content" not in data:
        return jsonify({"error": "Missing 'question' or 'content'"}), 400

    if not llm_service.is_loaded():
        return jsonify({"error": "No LLM loaded."}), 503

    try:
        result = query_scraped_content(
            question=data["question"],
            url=data.get("url", ""),
            title=data.get("title", ""),
            content=data["content"],
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/status", methods=["GET"])
def chat_status():
    """Check if chat services are ready."""
    from app.services.embedding import embedding_service
    from app.services.qdrant_store import qdrant_store

    return jsonify({
        "llm_loaded": llm_service.is_loaded(),
        "llm_info": llm_service.get_info(),
        "embedding_loaded": embedding_service.is_loaded(),
        "embedding_info": embedding_service.get_info(),
        "qdrant_connected": qdrant_store.is_connected(),
    })
