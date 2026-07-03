"""
Embeddings via an OpenAI-compatible API - no model runs on this machine.

Activated only when EMBEDDINGS_API_KEY is set. Works with any provider that
exposes POST /v1/embeddings (xAI when they ship one, OpenAI, Mistral, Jina,
Voyage...). Point EMBEDDINGS_API_BASE / EMBEDDINGS_MODEL at the provider.
"""

import logging
import threading
from typing import List

from app.config import embeddings_api_cfg

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._client = None
        return cls._instance

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=embeddings_api_cfg.api_key,
                base_url=embeddings_api_cfg.base_url,
            )
        return self._client

    def is_enabled(self) -> bool:
        return embeddings_api_cfg.enabled

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not embeddings_api_cfg.enabled:
            raise RuntimeError("Embeddings API not configured (EMBEDDINGS_API_KEY unset)")
        client = self._get_client()
        vectors: List[List[float]] = []
        bs = embeddings_api_cfg.batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            resp = client.embeddings.create(model=embeddings_api_cfg.model, input=batch)
            vectors.extend(d.embedding for d in resp.data)
        return vectors

    def get_info(self) -> dict:
        return {
            "enabled": embeddings_api_cfg.enabled,
            "provider": embeddings_api_cfg.base_url,
            "model": embeddings_api_cfg.model if embeddings_api_cfg.enabled else None,
            "mode": "api",
        }


embeddings_client = EmbeddingsClient()
