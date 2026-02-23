import os
import multiprocessing
from dataclasses import dataclass, field


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
    # Leave 2 cores for embedding, qdrant, system
    available = multiprocessing.cpu_count() or 4
    return max(1, available - 2)


@dataclass
class LLMConfig:
    model_path: str = os.environ.get("MODEL_PATH", "./models/phi-4-Q4_K_M.gguf")
    n_ctx: int = int(os.environ.get("MODEL_N_CTX", "4096"))
    n_threads: int = _auto_threads()
    n_batch: int = int(os.environ.get("MODEL_N_BATCH", "512"))
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
    top_k: int = 3  # keep top 3 after reranking


@dataclass
class VisionConfig:
    model_name: str = os.environ.get("VISION_MODEL", "openbmb/MiniCPM-V-2_6-int4")
    device: str = "cpu"
    max_image_size: int = 1344
    enabled: bool = os.environ.get("VISION_ENABLED", "0") == "1"


@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 8  # retrieve more, reranker narrows down
    score_threshold: float = 0.45
    rerank: bool = True  # cross-encoder reranking enabled


@dataclass
class ScraperConfig:
    timeout: int = 15
    max_content_length: int = 500_000  # ~500 KB
    user_agent: str = "MIRROR-Bot/1.0"


# Singleton configs
infra = InfraConfig()
llm_cfg = LLMConfig()
embedding_cfg = EmbeddingConfig()
qdrant_cfg = QdrantConfig()
rag_cfg = RAGConfig()
scraper_cfg = ScraperConfig()
reranker_cfg = RerankerConfig()
vision_cfg = VisionConfig()
