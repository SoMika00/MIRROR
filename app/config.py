import os
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class InfraConfig:
    """Hardware constraints: 64 GB RAM, 12 CPU cores, no GPU."""
    total_ram_gb: int = 64
    cpu_cores: int = 12
    # Budget allocation:
    #   LLM (Phi-4 Q4_K_M ~9 GB)  → 12 GB with KV cache
    #   Embedding (BGE-M3 ~1.2 GB) →  2 GB
    #   Qdrant                      →  4 GB (scales with collection size)
    #   OS + Flask + overhead       →  6 GB
    #   Free headroom               → 40 GB
    llm_ram_budget_gb: int = 12
    embedding_ram_budget_gb: int = 2
    qdrant_ram_budget_gb: int = 4


def _auto_threads() -> int:
    """Auto-detect available CPU cores. Let Docker cgroup limits apply naturally."""
    n = int(os.environ.get("MODEL_N_THREADS", "0"))
    if n > 0:
        return n
    # Prefer cpuset/affinity-aware detection so Docker/Compose CPU limits are respected.
    available = 0
    try:
        # Linux: returns CPUs the current process is allowed to run on (respects cpuset)
        available = len(os.sched_getaffinity(0))
    except Exception:
        available = 0

    if available <= 0:
        available = multiprocessing.cpu_count() or 4

    # Use all available cores for maximum inference throughput.
    # Flask/embedding/Qdrant run in separate threads and won't contend significantly.
    return max(1, available)


@dataclass
class LLMConfig:
    model_path: str = os.environ.get("MODEL_PATH", "./models/phi-4-Q4_K_M.gguf")
    n_ctx: int = int(os.environ.get("MODEL_N_CTX", "8192"))
    n_threads: int = _auto_threads()
    n_batch: int = int(os.environ.get("MODEL_N_BATCH", "1024"))
    use_mlock: bool = os.environ.get("MODEL_MLOCK", "1") == "1"
    use_mmap: bool = os.environ.get("MODEL_MMAP", "1") == "1"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    repeat_penalty: float = 1.1


@dataclass
class EmbeddingConfig:
    model_name: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    device: str = os.environ.get("EMBEDDING_DEVICE", "cpu")
    max_seq_length: int = 512
    batch_size: int = 8
    normalize: bool = True


@dataclass
class QdrantConfig:
    host: str = os.environ.get("QDRANT_HOST", "localhost")
    port: int = int(os.environ.get("QDRANT_PORT", "6333"))
    collection_name: str = os.environ.get("QDRANT_COLLECTION", "mirror_docs")
    vector_size: int = 1024  # BGE-M3 dense vector dimension
    # HNSW tuning for 12-core CPU
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    hnsw_ef_search: int = 128
    # Quantization
    quantization: str = "int8"
    always_ram: bool = True


@dataclass
class RerankerConfig:
    model_name: str = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    device: str = "cpu"
    top_k: int = 5  # keep top 5 after reranking (8192 ctx can handle more)


@dataclass
class VisionConfig:
    model_name: str = os.environ.get("VISION_MODEL", "openbmb/MiniCPM-V-2_6-int4")
    device: str = "cpu"
    max_image_size: int = 1344
    enabled: bool = os.environ.get("VISION_ENABLED", "0") == "1"


@dataclass
class RAGConfig:
    chunk_size: int = 768
    chunk_overlap: int = 128
    top_k: int = 12  # retrieve more, reranker narrows down
    score_threshold: float = 0.35
    rerank: bool = True  # cross-encoder reranking enabled


@dataclass
class ScraperConfig:
    timeout: int = 15
    max_content_length: int = 500_000  # ~500 KB
    user_agent: str = "MIRROR-Bot/1.0"


# ---------------------------------------------------------------------------
# Model Registry - all supported LLM models with HuggingFace download info
# ---------------------------------------------------------------------------

MODEL_REGISTRY: List[Dict] = [
    # --- Phi-4-mini 3.8B (default - fast inference) ---
    {
        "id": "phi-4-mini-q4km",
        "name": "Phi-4 Mini 3.8B Q4_K_M",
        "family": "Phi-4 Mini",
        "params": "3.8B",
        "quant": "Q4_K_M",
        "ram_gb": 3,
        "filename": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "hf_repo": "unsloth/Phi-4-mini-instruct-GGUF",
        "hf_file": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "description": "Fast, compact. Great reasoning for its size. Ideal for CPU.",
        "n_ctx": 8192,
        "speed_estimate": "15-25 t/s",
        "default": False,
    },
    # --- Phi-4 14B ---
    {
        "id": "phi-4-14b-q8",
        "name": "Phi-4 14B Q8_0",
        "family": "Phi-4",
        "params": "14B",
        "quant": "Q8_0",
        "ram_gb": 16,
        "filename": "phi-4-Q8_0.gguf",
        "hf_repo": "bartowski/phi-4-GGUF",
        "hf_file": "phi-4-Q8_0.gguf",
        "description": "High quality 14B. Slow on CPU (3-6 t/s).",
        "n_ctx": 8192,
        "speed_estimate": "3-6 t/s",
        "default": False,
    },
    {
        "id": "phi-4-14b-q4km",
        "name": "Phi-4 14B Q4_K_M",
        "family": "Phi-4",
        "params": "14B",
        "quant": "Q4_K_M",
        "ram_gb": 9,
        "filename": "phi-4-Q4_K_M.gguf",
        "hf_repo": "bartowski/phi-4-GGUF",
        "hf_file": "phi-4-Q4_K_M.gguf",
        "description": "Phi-4 lighter quantization. Faster than Q8.",
        "n_ctx": 8192,
        "speed_estimate": "5-10 t/s",
        "default": False,
    },
    # --- Qwen 2.5 32B ---
    {
        "id": "qwen2.5-32b-q6k",
        "name": "Qwen 2.5 32B Q6_K",
        "family": "Qwen 2.5",
        "params": "32B",
        "quant": "Q6_K",
        "ram_gb": 28,
        "filename": "Qwen2.5-32B-Instruct-Q6_K.gguf",
        "hf_repo": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "hf_file": "Qwen2.5-32B-Instruct-Q6_K.gguf",
        "description": "Top-tier multilingual. Excellent reasoning & code generation.",
        "n_ctx": 8192,
        "speed_estimate": "1-3 t/s",
        "default": False,
    },
    {
        "id": "qwen2.5-32b-q4km",
        "name": "Qwen 2.5 32B Q4_K_M",
        "family": "Qwen 2.5",
        "params": "32B",
        "quant": "Q4_K_M",
        "ram_gb": 20,
        "filename": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "hf_repo": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "hf_file": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "description": "Qwen 32B lighter quantization. Faster inference, still excellent quality.",
        "n_ctx": 8192,
        "speed_estimate": "2-5 t/s",
        "default": False,
    },
    # --- Phi-3.5 MoE 42B ---
    {
        "id": "phi3.5-moe-q8",
        "name": "Phi-3.5 MoE 42B Q8_0",
        "family": "Phi-3.5 MoE",
        "params": "42B",
        "quant": "Q8_0",
        "ram_gb": 46,
        "filename": "Phi-3.5-MoE-instruct-Q8_0.gguf",
        "hf_repo": "bartowski/Phi-3.5-MoE-instruct-GGUF",
        "hf_file": "Phi-3.5-MoE-instruct-Q8_0.gguf",
        "description": "MoE architecture - activates 6.6B of 42B per token. Complex reasoning.",
        "n_ctx": 8192,
        "speed_estimate": "1-2 t/s",
        "default": False,
    },
    # --- Llama 3.2 8B (multiple quantizations for comparison) ---
    {
        "id": "llama3.2-8b-fp16",
        "name": "Llama 3.1 8B FP16",
        "family": "Llama 3",
        "params": "8B",
        "quant": "FP16",
        "ram_gb": 16,
        "filename": "Meta-Llama-3.1-8B-Instruct-f16.gguf",
        "hf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "hf_file": "Meta-Llama-3.1-8B-Instruct-f16.gguf",
        "description": "Full precision baseline for quality comparison.",
        "n_ctx": 8192,
        "speed_estimate": "4-8 t/s",
        "default": False,
    },
    {
        "id": "llama3.2-8b-q8",
        "name": "Llama 3.1 8B Q8_0",
        "family": "Llama 3",
        "params": "8B",
        "quant": "Q8_0",
        "ram_gb": 8,
        "filename": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        "hf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "hf_file": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        "description": "Near-lossless quantization. Best quality/speed trade-off.",
        "n_ctx": 8192,
        "speed_estimate": "6-12 t/s",
        "default": False,
    },
    {
        "id": "llama3.2-8b-q6k",
        "name": "Llama 3.1 8B Q6_K",
        "family": "Llama 3",
        "params": "8B",
        "quant": "Q6_K",
        "ram_gb": 7,
        "filename": "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
        "hf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "hf_file": "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
        "description": "Balanced quantization. Minimal quality loss. Best default for CPU.",
        "n_ctx": 8192,
        "speed_estimate": "8-14 t/s",
        "default": True,
    },
    {
        "id": "llama3.2-8b-q4km",
        "name": "Llama 3.1 8B Q4_K_M",
        "family": "Llama 3",
        "params": "8B",
        "quant": "Q4_K_M",
        "ram_gb": 5,
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "hf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "hf_file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "description": "Aggressive quantization. Fastest, see quality impact.",
        "n_ctx": 8192,
        "speed_estimate": "10-18 t/s",
        "default": False,
    },
]


def get_model_by_id(model_id: str) -> dict:
    """Look up a model entry from the registry by its ID."""
    for m in MODEL_REGISTRY:
        if m["id"] == model_id:
            return m
    return {}


def get_default_model() -> dict:
    """Return the default model entry."""
    for m in MODEL_REGISTRY:
        if m.get("default"):
            return m
    return MODEL_REGISTRY[0] if MODEL_REGISTRY else {}


# Singleton configs
infra = InfraConfig()
llm_cfg = LLMConfig()
embedding_cfg = EmbeddingConfig()
qdrant_cfg = QdrantConfig()
rag_cfg = RAGConfig()
scraper_cfg = ScraperConfig()
reranker_cfg = RerankerConfig()
vision_cfg = VisionConfig()
