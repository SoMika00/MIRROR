"""Model management routes — load/unload/switch LLM and embedding models."""

import os
import threading
import logging
from flask import Blueprint, request, jsonify

from app.services.llm import llm_service
from app.services.embedding import embedding_service
from app.services.reranker import reranker_service
from app.services.vision import vision_service
from app.services.qdrant_store import qdrant_store
from app.config import MODEL_REGISTRY, get_model_by_id, get_default_model

logger = logging.getLogger(__name__)
models_bp = Blueprint("models", __name__)

# In-memory download progress tracking
_download_progress = {}  # model_id -> {"status": str, "progress": float, "error": str|None}

# In-memory load/unload status tracking
_load_status = {"status": "idle", "model_id": None, "model_name": None, "step": "", "error": None}


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
    """Load or switch LLM model. Non-blocking — launches in background thread."""
    global _load_status
    data = request.get_json() or {}
    model_path = data.get("model_path")
    model_id = data.get("model_id")

    if _load_status["status"] == "loading":
        return jsonify({"success": True, "message": "Load already in progress"})

    # Resolve name for UI
    model_name = model_id or (model_path.split("/")[-1] if model_path else "default")
    if model_id:
        entry = get_model_by_id(model_id)
        if entry:
            model_name = entry["name"]

    _load_status = {"status": "loading", "model_id": model_id, "model_name": model_name, "step": "Preparing...", "error": None}

    def _do_load():
        global _load_status
        try:
            if llm_service.is_loaded():
                _load_status["step"] = "Unloading current model..."
                llm_service.unload()
            _load_status["step"] = f"Loading {model_name} into RAM..."
            llm_service.load(model_path=model_path, model_id=model_id)
            _load_status = {"status": "done", "model_id": model_id, "model_name": model_name, "step": "Ready", "error": None}
        except Exception as e:
            logger.error(f"Load failed for {model_id}: {e}")
            _load_status = {"status": "error", "model_id": model_id, "model_name": model_name, "step": "", "error": str(e)}

    thread = threading.Thread(target=_do_load, daemon=True)
    thread.start()
    return jsonify({"success": True, "message": "Load started"})


@models_bp.route("/llm/unload", methods=["POST"])
def unload_llm():
    """Unload current LLM to free RAM."""
    global _load_status
    _load_status = {"status": "loading", "model_id": None, "model_name": "current model", "step": "Unloading...", "error": None}
    llm_service.unload()
    _load_status = {"status": "idle", "model_id": None, "model_name": None, "step": "", "error": None}
    return jsonify({"success": True})


@models_bp.route("/llm/load-status", methods=["GET"])
def load_status():
    """Poll model load/unload progress."""
    return jsonify(_load_status)


@models_bp.route("/llm/list", methods=["GET"])
def list_llms():
    """List available GGUF models in ./models/."""
    models = llm_service.list_available_models()
    return jsonify({"models": models})


@models_bp.route("/llm/registry", methods=["GET"])
def list_registry():
    """Return the full model registry with download/active status."""
    return jsonify({"models": llm_service.list_registry_models()})


@models_bp.route("/llm/download", methods=["POST"])
def download_model():
    """Download a model from HuggingFace by model_id. Non-blocking."""
    data = request.get_json() or {}
    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400

    entry = get_model_by_id(model_id)
    if not entry:
        return jsonify({"error": f"Unknown model_id: {model_id}"}), 404

    dest = f"./models/{entry['filename']}"
    if os.path.exists(dest):
        return jsonify({"success": True, "message": "Already downloaded", "path": dest})

    # Check if download already in progress
    if model_id in _download_progress and _download_progress[model_id]["status"] == "downloading":
        return jsonify({"success": True, "message": "Download already in progress"})

    _download_progress[model_id] = {"status": "downloading", "progress": 0.0, "error": None}

    def _do_download():
        try:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_token")
            from huggingface_hub import hf_hub_download
            logger.info(f"Downloading {entry['hf_repo']}/{entry['hf_file']} -> {dest}")
            _download_progress[model_id]["progress"] = 0.1
            hf_hub_download(
                repo_id=entry["hf_repo"],
                filename=entry["hf_file"],
                local_dir="./models",
                local_dir_use_symlinks=False,
                token=hf_token,
            )
            # Rename if the downloaded file name differs from our expected filename
            downloaded_path = f"./models/{entry['hf_file']}"
            if downloaded_path != dest and os.path.exists(downloaded_path):
                os.rename(downloaded_path, dest)
            _download_progress[model_id] = {"status": "done", "progress": 1.0, "error": None}
            logger.info(f"Download complete: {dest}")
        except Exception as e:
            logger.error(f"Download failed for {model_id}: {e}")
            _download_progress[model_id] = {"status": "error", "progress": 0.0, "error": str(e)}

    thread = threading.Thread(target=_do_download, daemon=True)
    thread.start()
    return jsonify({"success": True, "message": "Download started"})


@models_bp.route("/llm/download-progress/<model_id>", methods=["GET"])
def download_progress(model_id):
    """Check download progress for a model."""
    # Also check if file now exists (covers manual downloads)
    entry = get_model_by_id(model_id)
    if entry and os.path.exists(f"./models/{entry['filename']}"):
        return jsonify({"status": "done", "progress": 1.0, "error": None})
    prog = _download_progress.get(model_id, {"status": "idle", "progress": 0.0, "error": None})
    return jsonify(prog)


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
    data = request.get_json() or {}
    model_id = data.get("model_id")
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
        if llm_service.is_loaded():
            llm_service.unload()
        # Use provided model_id, or MODEL_PATH env, or registry default
        if model_id:
            llm_service.load(model_id=model_id)
        else:
            env_path = os.environ.get("MODEL_PATH", "")
            if env_path and os.path.exists(env_path):
                llm_service.load(model_path=env_path)
            else:
                default = get_default_model()
                if default:
                    llm_service.load(model_id=default["id"])
                else:
                    raise RuntimeError("No model available to load")
        results["llm"] = "loaded"
    except Exception as e:
        results["llm"] = f"error: {e}"
    return jsonify({"results": results})
