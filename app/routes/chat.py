"""Chat API routes — supports multiple modes: chat, rag, scrap, fulldoc."""

import json
import time
import logging
from flask import Blueprint, request, jsonify, Response, stream_with_context, make_response

from app.services.rag import (
    query_rag, query_rag_stream, query_scraped_content,
    query_direct_chat, query_direct_chat_stream,
)
from app.services.llm import llm_service
from app.services.database import db
from app.services.query_router import classify_query

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__)


def _get_user_id() -> str:
    """Get or create user from cookie."""
    user_id = request.cookies.get("mirror_uid")
    return db.get_or_create_user(user_id)


def _set_user_cookie(response, user_id: str):
    """Set persistent user cookie (1 year)."""
    response.set_cookie("mirror_uid", user_id, max_age=365 * 24 * 3600,
                         httponly=True, samesite="Lax")
    return response


# --- Conversations ---

@chat_bp.route("/conversations", methods=["GET"])
def list_conversations():
    user_id = _get_user_id()
    convs = db.get_conversations(user_id)
    resp = make_response(jsonify({"conversations": convs}))
    return _set_user_cookie(resp, user_id)


@chat_bp.route("/conversations", methods=["POST"])
def create_conversation():
    user_id = _get_user_id()
    data = request.get_json() or {}
    mode = data.get("mode", "chat")
    title = data.get("title", "New conversation")
    conv_id = db.create_conversation(user_id, mode=mode, title=title)
    resp = make_response(jsonify({"conversation_id": conv_id}))
    return _set_user_cookie(resp, user_id)


@chat_bp.route("/conversations/<conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    user_id = _get_user_id()
    ok = db.delete_conversation(conv_id, user_id)
    return jsonify({"deleted": ok})


@chat_bp.route("/conversations/<conv_id>/messages", methods=["GET"])
def get_messages(conv_id):
    messages = db.get_messages(conv_id)
    return jsonify({"messages": messages})


# --- Chat Query (all modes) ---

@chat_bp.route("/query", methods=["POST"])
def chat_query():
    """Unified query endpoint. Modes: chat, rag, scrap, fulldoc."""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()
    mode = data.get("mode", "chat")
    conv_id = data.get("conversation_id")
    source_type = data.get("source_type")
    enabled_sources = data.get("enabled_sources")

    if not question:
        return jsonify({"error": "Empty question"}), 400

    if not llm_service.is_loaded():
        return jsonify({"error": "No LLM loaded. Please load a model first via the model manager."}), 503

    user_id = _get_user_id()

    # Auto-create conversation if needed
    if not conv_id:
        conv_id = db.create_conversation(user_id, mode=mode, title=question[:60])

    # Save user message
    db.add_message(conv_id, "user", question, mode=mode)

    # Log
    db.add_log("info", "chat", f"Query [{mode}]: {question[:80]}", user_id=user_id)

    try:
        history = db.get_recent_context(conv_id, n=6)

        # Adaptive routing: classify query complexity
        has_sources = bool(enabled_sources) or mode in ("rag", "fulldoc")
        route = classify_query(question, has_sources=has_sources)
        logger.info(f"Route: {route.tier}/{route.mode} — {route.reason}")

        # Auto-upgrade mode based on router if user didn't explicitly choose
        effective_mode = mode
        if mode == "chat" and route.mode == "rag" and has_sources:
            effective_mode = "rag"
        elif mode == "rag" and route.mode == "chat" and not enabled_sources:
            effective_mode = "chat"

        if effective_mode == "rag":
            result = query_rag(question, source_type=source_type, enabled_sources=enabled_sources)
        elif effective_mode == "scrap":
            content = data.get("content", "")
            url = data.get("url", "")
            title_page = data.get("title", "")
            if not content:
                result = query_direct_chat(question, history=history)
            else:
                result = query_scraped_content(question, url, title_page, content)
        elif effective_mode == "fulldoc":
            result = query_rag(question, source_type="document", enabled_sources=enabled_sources)
        else:
            result = query_direct_chat(question, history=history)

        # Attach routing metadata
        result["route"] = {"tier": route.tier, "mode": effective_mode, "reason": route.reason}

        # Save assistant response
        db.add_message(conv_id, "assistant", result["answer"], mode=mode,
                       sources=result.get("sources"), timings=result.get("timings"))

        # Auto-title: use first question
        result["conversation_id"] = conv_id

        resp = make_response(jsonify(result))
        return _set_user_cookie(resp, user_id)

    except Exception as e:
        logger.error(f"Chat query failed: {e}", exc_info=True)
        db.add_log("error", "chat", f"Query failed: {e}", user_id=user_id)
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/stream", methods=["POST"])
def chat_stream():
    """Streaming query (all modes)."""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()
    mode = data.get("mode", "chat")
    conv_id = data.get("conversation_id")
    source_type = data.get("source_type")
    enabled_sources = data.get("enabled_sources")

    if not llm_service.is_loaded():
        return jsonify({"error": "No LLM loaded."}), 503

    user_id = _get_user_id()

    if not conv_id:
        conv_id = db.create_conversation(user_id, mode=mode, title=question[:60])

    db.add_message(conv_id, "user", question, mode=mode)
    history = db.get_recent_context(conv_id, n=6)

    def generate():
        full_answer = []
        sources_data = []
        try:
            if mode in ("rag", "fulldoc"):
                st = "document" if mode == "fulldoc" else source_type
                stream = query_rag_stream(question, source_type=st, enabled_sources=enabled_sources)
                # query_rag_stream yields either tokens (str) or a sources dict
                for item in stream:
                    if isinstance(item, dict) and 'sources' in item:
                        sources_data = item['sources']
                        yield f"data: {json.dumps({'sources': sources_data})}\n\n"
                    else:
                        full_answer.append(item)
                        yield f"data: {json.dumps({'token': item})}\n\n"
            else:
                stream = query_direct_chat_stream(question, history=history)
                for token in stream:
                    full_answer.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # Save full answer
            answer_text = "".join(full_answer)
            db.add_message(conv_id, "assistant", answer_text, mode=mode,
                           sources=sources_data if sources_data else None)

            yield f"data: {json.dumps({'done': True, 'conversation_id': conv_id, 'sources': sources_data})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Scraper Query (legacy compat) ---

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


# --- Sources ---

@chat_bp.route("/sources", methods=["GET"])
def list_sources():
    user_id = _get_user_id()
    sources = db.get_user_sources(user_id)
    resp = make_response(jsonify({"sources": sources}))
    return _set_user_cookie(resp, user_id)


@chat_bp.route("/sources/<source_id>", methods=["DELETE"])
def delete_source(source_id):
    user_id = _get_user_id()
    source_name = db.delete_user_source(source_id, user_id)
    if source_name:
        try:
            from app.services.qdrant_store import qdrant_store
            qdrant_store.delete_by_source(source_name)
        except Exception as e:
            logger.warning(f"Failed to delete from Qdrant: {e}")
        return jsonify({"deleted": True, "source_name": source_name})
    return jsonify({"deleted": False}), 404


# --- Status ---

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


# --- Logs ---

@chat_bp.route("/logs", methods=["GET"])
def get_logs():
    limit = request.args.get("limit", 100, type=int)
    component = request.args.get("component")
    level = request.args.get("level")
    logs = db.get_logs(limit=limit, component=component, level=level)
    return jsonify({"logs": logs})
