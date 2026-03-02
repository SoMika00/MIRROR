"""Model management routes - load/unload/switch LLM and embedding models."""

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


# ---------------------------------------------------------------------------
# Real-time CPU measurement state (differential /proc/stat sampling)
# ---------------------------------------------------------------------------
_prev_cpu_times = None
_prev_cpu_timestamp = 0.0
_cpu_lock = threading.Lock()

# Inference telemetry - updated live during generation
inference_telemetry = {
    "active": False,
    "tokens_generated": 0,
    "tokens_per_sec": 0.0,
    "start_time": 0.0,
    "model_name": "",
}


def _read_proc_stat():
    """Read aggregate CPU times from /proc/stat (jiffies)."""
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            line = f.readline()  # first line: cpu  user nice system idle iowait irq softirq steal
        parts = line.split()
        if parts[0] != "cpu":
            return None
        times = [int(x) for x in parts[1:]]
        idle = times[3] + (times[4] if len(times) > 4 else 0)  # idle + iowait
        total = sum(times)
        return {"idle": idle, "total": total}
    except Exception:
        return None


def _read_per_core_stat():
    """Read per-core CPU usage from /proc/stat."""
    cores = []
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("cpu") and line[3] != " ":
                    parts = line.split()
                    times = [int(x) for x in parts[1:]]
                    idle = times[3] + (times[4] if len(times) > 4 else 0)
                    total = sum(times)
                    cores.append({"idle": idle, "total": total})
    except Exception:
        pass
    return cores


_prev_per_core = []
_prev_per_core_ts = 0.0


@models_bp.route("/metrics", methods=["GET"])
def metrics():
    import time as _time
    global _prev_cpu_times, _prev_cpu_timestamp, _prev_per_core, _prev_per_core_ts

    # --- RAM (from /proc/meminfo - always real-time) ---
    def _read_meminfo():
        total_kb = None
        avail_kb = None
        buffers_kb = 0
        cached_kb = 0
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                    elif line.startswith("Buffers:"):
                        buffers_kb = int(line.split()[1])
                    elif line.startswith("Cached:"):
                        cached_kb = int(line.split()[1])
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
            "buffers_cached_gb": round((buffers_kb + cached_kb) / 1024 / 1024, 2),
        }

    # --- CPU (real-time differential from /proc/stat) ---
    def _cpu_percent_realtime():
        global _prev_cpu_times, _prev_cpu_timestamp
        now = _time.monotonic()
        current = _read_proc_stat()
        if current is None:
            return None

        with _cpu_lock:
            if _prev_cpu_times is None or (now - _prev_cpu_timestamp) < 0.05:
                _prev_cpu_times = current
                _prev_cpu_timestamp = now
                # First call: fall back to load average for initial value
                try:
                    load1, _, _ = os.getloadavg()
                    n = os.cpu_count() or 1
                    return round(min(100.0, (load1 / n) * 100.0), 1)
                except Exception:
                    return 0.0

            d_total = current["total"] - _prev_cpu_times["total"]
            d_idle = current["idle"] - _prev_cpu_times["idle"]
            _prev_cpu_times = current
            _prev_cpu_timestamp = now

        if d_total <= 0:
            return 0.0
        return round(((d_total - d_idle) / d_total) * 100.0, 1)

    # --- Per-core CPU ---
    def _per_core_percent():
        global _prev_per_core, _prev_per_core_ts
        now = _time.monotonic()
        current = _read_per_core_stat()
        if not current:
            return []

        if not _prev_per_core or len(_prev_per_core) != len(current):
            _prev_per_core = current
            _prev_per_core_ts = now
            return [0.0] * len(current)

        percents = []
        for prev, cur in zip(_prev_per_core, current):
            dt = cur["total"] - prev["total"]
            di = cur["idle"] - prev["idle"]
            if dt <= 0:
                percents.append(0.0)
            else:
                percents.append(round(((dt - di) / dt) * 100.0, 1))
        _prev_per_core = current
        _prev_per_core_ts = now
        return percents

    # --- GPU ---
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

    # --- Process-level memory breakdown ---
    def _process_memory():
        """Read RSS of current process from /proc/self/status."""
        rss_kb = 0
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
        except Exception:
            pass
        return {"rss_gb": round(rss_kb / 1024 / 1024, 2), "rss_kb": rss_kb}

    ram_info = _read_meminfo()
    proc_info = _process_memory()

    # Fix misleading RAM %: mmap'd model files are counted as "cache" by Linux,
    # so MemAvailable stays high even when the process uses 40+ GB.
    # Use max(system_used%, process_rss%) as the displayed percentage.
    if ram_info and proc_info.get("rss_kb"):
        total_kb = ram_info.get("total_gb", 0) * 1024 * 1024
        if total_kb > 0:
            rss_pct = (proc_info["rss_kb"] / total_kb) * 100.0
            ram_info["percent"] = round(max(ram_info["percent"], rss_pct), 1)
            ram_info["used_gb"] = round(max(ram_info["used_gb"], proc_info["rss_gb"]), 2)

    return jsonify({
        "cpu_percent": _cpu_percent_realtime(),
        "cpu_per_core": _per_core_percent(),
        "ram": ram_info,
        "process": proc_info,
        "gpu": _gpu_metrics(),
        "inference": inference_telemetry,
    })


@models_bp.route("/llm/load", methods=["POST"])
def load_llm():
    """Load or switch LLM model. Non-blocking - launches in background thread."""
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
