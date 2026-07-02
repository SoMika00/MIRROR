"""Service status and budget routes."""

import logging
from flask import Blueprint, jsonify

from app.services.llm import llm_service
from app.services.embedding import embedding_service
from app.services.reranker import reranker_service
from app.services.qdrant_store import qdrant_store

logger = logging.getLogger(__name__)
models_bp = Blueprint("models", __name__)


@models_bp.route("/status", methods=["GET"])
def models_status():
    """Get status of all services."""
    qdrant_info = {"connected": False}
    try:
        qdrant_info = qdrant_store.get_collection_info()
    except Exception:
        pass

    return jsonify({
        "llm": llm_service.get_info(),
        "embedding": embedding_service.get_info(),
        "reranker": reranker_service.get_info(),
        "qdrant": qdrant_info,
    })


@models_bp.route("/budget", methods=["GET"])
def budget_status():
    """Get current daily token budget usage."""
    return jsonify(llm_service.get_info()["daily_budget"])


@models_bp.route("/embedding/load", methods=["POST"])
def load_embedding():
    """Load the embedding model."""
    try:
        embedding_service.load()
        return jsonify({"success": True, "info": embedding_service.get_info()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/qdrant/connect", methods=["POST"])
def connect_qdrant():
    """Connect to Qdrant."""
    try:
        qdrant_store.connect()
        return jsonify({"success": True, "info": qdrant_store.get_collection_info()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/reranker/load", methods=["POST"])
def load_reranker():
    """Load the reranker model."""
    try:
        reranker_service.load()
        return jsonify({"success": True, "info": reranker_service.get_info()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/load-all", methods=["POST"])
def load_all():
    """Load all services."""
    results = {}
    try:
        qdrant_store.connect()
        results["qdrant"] = "connected"
    except Exception as e:
        results["qdrant"] = f"error: {e}"
    try:
        embedding_service.load()
        results["embedding"] = "loaded"
    except Exception as e:
        results["embedding"] = f"error: {e}"
    try:
        reranker_service.load()
        results["reranker"] = "loaded"
    except Exception as e:
        results["reranker"] = f"error: {e}"
    try:
        llm_service.load()
        results["llm"] = "ready (Grok API)"
    except Exception as e:
        results["llm"] = f"error: {e}"
    return jsonify({"results": results})
