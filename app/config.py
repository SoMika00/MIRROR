import os
from dataclasses import dataclass


@dataclass
class GrokConfig:
    """xAI Grok API configuration (chat completions)."""
    api_key: str = os.environ.get("GROK_API_KEY", "")
    model: str = os.environ.get("GROK_MODEL", "grok-4.20-non-reasoning")
    base_url: str = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")
    daily_budget_usd: float = float(os.environ.get("GROK_DAILY_BUDGET", "0.50"))
    # Pricing in USD per 1M tokens (grok-4.x family)
    input_price_per_m: float = float(os.environ.get("GROK_INPUT_PRICE", "1.25"))
    output_price_per_m: float = float(os.environ.get("GROK_OUTPUT_PRICE", "2.50"))
    max_tokens: int = 1024
    temperature: float = 0.7


@dataclass
class EmbeddingsAPIConfig:
    """Optional OpenAI-compatible embeddings API (hybrid retrieval).

    Disabled unless EMBEDDINGS_API_KEY is set. xAI does not expose an
    embeddings endpoint yet; when it does (or with any other provider),
    set base_url/model/key and hybrid search activates automatically.
    """
    api_key: str = os.environ.get("EMBEDDINGS_API_KEY", "")
    base_url: str = os.environ.get("EMBEDDINGS_API_BASE", "https://api.x.ai/v1")
    model: str = os.environ.get("EMBEDDINGS_MODEL", "grok-embedding")
    batch_size: int = 64

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass
class RAGConfig:
    chunk_size: int = 768
    chunk_overlap: int = 128
    top_k: int = 8
    # Weight of vector score vs BM25 when hybrid search is active
    hybrid_vector_weight: float = 0.6


@dataclass
class ScraperConfig:
    timeout: int = 15
    max_content_length: int = 500_000
    user_agent: str = "MIRROR-Bot/1.0"


# Singleton configs
grok_cfg = GrokConfig()
embeddings_api_cfg = EmbeddingsAPIConfig()
rag_cfg = RAGConfig()
scraper_cfg = ScraperConfig()
