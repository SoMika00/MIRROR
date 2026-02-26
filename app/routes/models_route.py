"""Model management routes — load/unload/switch LLM and embedding models."""

import os
import threading
import logging
import subprocess
import shutil
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


@models_bp.route("/metrics", methods=["GET"])
def metrics():
    def _read_meminfo():
        total_kb = None
        avail_kb = None
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
        except Exception:
            return None
        if not total_kb or avail_kb is None:
            return None
        used_kb = max(0, total_kb - avail_kb)
        pct = (used_kb / total_kb) * 100.0 if total_kb else 0.0
        return {
            "total_gb": round(total_kb / 1024 / 1024, 2),
            "used_gb": round(used_kb / 1024 / 1024, 2),
            "percent": round(pct, 1),
        }

    def _cpu_percent_estimate():
        try:
            load1, _, _ = os.getloadavg()
            n = os.cpu_count() or 1
            return round(min(100.0, (load1 / n) * 100.0), 1)
        except Exception:
            return None

    def _gpu_metrics():
        if not shutil.which("nvidia-smi"):
            return None
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            ).strip()
            if not out:
                return None
            first = out.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) < 4:
                return None
            gpu_util = float(parts[0])
            mem_util = float(parts[1])
            mem_used = float(parts[2])
            mem_total = float(parts[3])
            mem_pct = (mem_used / mem_total) * 100.0 if mem_total else 0.0
            return {
                "gpu_percent": round(gpu_util, 1),
                "mem_percent": round(mem_pct, 1),
                "mem_used_mb": round(mem_used, 0),
                "mem_total_mb": round(mem_total, 0),
                "mem_util_percent": round(mem_util, 1),
            }
        except Exception:
            return None

    return jsonify({
        "cpu_percent": _cpu_percent_estimate(),
        "ram": _read_meminfo(),
        "gpu": _gpu_metrics(),
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
    return jsonify({"error": "Model download is disabled. Place model files in the ./models volume."}), 403


@models_bp.route("/llm/download-progress/<model_id>", methods=["GET"])
def download_progress(model_id):
    """Check download progress for a model."""
    return jsonify({"error": "Model download is disabled. Place model files in the ./models volume."}), 403


@models_bp.route("/llm/download-all", methods=["POST"])
def download_all_models():
    return jsonify({"error": "Model download is disabled. Place model files in the ./models volume."}), 403


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
