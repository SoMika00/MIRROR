import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GrokConfig:
    """xAI Grok API configuration."""
    api_key: str = os.environ.get("GROK_API_KEY", "")
    model: str = os.environ.get("GROK_MODEL", "grok-2")
    base_url: str = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")
    daily_budget_usd: float = float(os.environ.get("GROK_DAILY_BUDGET", "0.50"))
    max_tokens: int = 1024
    temperature: float = 0.7


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
    vector_size: int = 1024
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    hnsw_ef_search: int = 128
    quantization: str = "int8"
    always_ram: bool = True


@dataclass
class RerankerConfig:
    model_name: str = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    device: str = "cpu"
    top_k: int = 5


@dataclass
class RAGConfig:
    chunk_size: int = 768
    chunk_overlap: int = 128
    top_k: int = 12
    score_threshold: float = 0.35
    rerank: bool = True


@dataclass
class ScraperConfig:
    timeout: int = 15
    max_content_length: int = 500_000
    user_agent: str = "MIRROR-Bot/1.0"


# Singleton configs
grok_cfg = GrokConfig()
embedding_cfg = EmbeddingConfig()
qdrant_cfg = QdrantConfig()
rag_cfg = RAGConfig()
scraper_cfg = ScraperConfig()
reranker_cfg = RerankerConfig()
