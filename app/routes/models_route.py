"""Model management routes — load/unload/switch LLM and embedding models."""

from flask import Blueprint, request, jsonify

from app.services.llm import llm_service
from app.services.embedding import embedding_service
from app.services.reranker import reranker_service
from app.services.vision import vision_service
from app.services.qdrant_store import qdrant_store

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
        "vision": vision_service.get_info(),
        "qdrant": qdrant_info,
    })


@models_bp.route("/llm/load", methods=["POST"])
def load_llm():
    """Load or switch LLM model."""
    data = request.get_json() or {}
    model_path = data.get("model_path")

    try:
        if llm_service.is_loaded():
            llm_service.unload()
        llm_service.load(model_path)
        return jsonify({"success": True, "info": llm_service.get_info()})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/llm/unload", methods=["POST"])
def unload_llm():
    """Unload current LLM to free RAM."""
    llm_service.unload()
    return jsonify({"success": True})


@models_bp.route("/llm/list", methods=["GET"])
def list_llms():
    """List available GGUF models in ./models/."""
    models = llm_service.list_available_models()
    return jsonify({"models": models})


@models_bp.route("/llm/test", methods=["POST"])
def test_llm():
    """Quick inference test to measure speed."""
    if not llm_service.is_loaded():
        return jsonify({"error": "No LLM loaded"}), 503

    data = request.get_json() or {}
    prompt = data.get("prompt", "Hello, who are you?")

    import time
    start = time.time()
    response = llm_service.generate(prompt, max_tokens=100)
    elapsed = time.time() - start

    return jsonify({
        "response": response,
        "elapsed_seconds": round(elapsed, 2),
        "estimated_tokens": len(response.split()),
    })


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


@models_bp.route("/vision/load", methods=["POST"])
def load_vision():
    """Load the vision model."""
    try:
        vision_service.load()
        return jsonify({"success": True, "info": vision_service.get_info()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/vision/unload", methods=["POST"])
def unload_vision():
    """Unload the vision model to free RAM."""
    vision_service.unload()
    return jsonify({"success": True})


@models_bp.route("/load-all", methods=["POST"])
def load_all():
    """Load all services in the right order."""
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
        results["llm"] = "loaded"
    except Exception as e:
        results["llm"] = f"error: {e}"
    return jsonify({"results": results})
