"""Service status and budget routes (API-first stack: nothing to load/unload)."""

import logging
from flask import Blueprint, jsonify

from app.services.llm import llm_service
from app.services.retrieval import retrieval_store
from app.services.embeddings_api import embeddings_client

logger = logging.getLogger(__name__)
models_bp = Blueprint("models", __name__)


@models_bp.route("/status", methods=["GET"])
def models_status():
    """Get status of all services."""
    retrieval_info = {"connected": False}
    try:
        retrieval_info = retrieval_store.get_info()
    except Exception:
        pass

    return jsonify({
        "llm": llm_service.get_info(),
        "retrieval": retrieval_info,
        "embeddings": embeddings_client.get_info(),
    })


@models_bp.route("/budget", methods=["GET"])
def budget_status():
    """Get current daily token budget usage."""
    return jsonify(llm_service.get_info()["daily_budget"])
