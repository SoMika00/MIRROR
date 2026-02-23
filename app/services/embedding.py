"""
Embedding service using BGE-M3 via sentence-transformers.

BGE-M3 (567M params) chosen for:
  - Multilingual support (100+ languages incl. French, English, Japanese)
  - Dense + sparse + hybrid retrieval capabilities
  - 1024-dim dense vectors → good accuracy/speed tradeoff on CPU
  - <30ms per query on CPU (sentence-transformers benchmarks)
  - Apache 2.0 license
"""

import time
import logging
import threading
from typing import List, Optional

import numpy as np

from app.config import embedding_cfg

logger = logging.getLogger(__name__)


class EmbeddingService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.model_name = embedding_cfg.model_name
        self.device = embedding_cfg.device
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        start = time.time()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
        self.model.max_seq_length = embedding_cfg.max_seq_length
        elapsed = time.time() - start
        logger.info(f"Embedding model loaded in {elapsed:.1f}s")
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        if not self._loaded:
            self.load()
        start = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=embedding_cfg.batch_size,
            normalize_embeddings=embedding_cfg.normalize,
            show_progress_bar=show_progress,
        )
        elapsed = time.time() - start
        logger.debug(f"Encoded {len(texts)} texts in {elapsed:.2f}s ({elapsed/max(len(texts),1)*1000:.0f}ms/text)")
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]

    def get_info(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "loaded": self._loaded,
            "vector_dim": embedding_cfg.max_seq_length,
            "max_seq_length": embedding_cfg.max_seq_length,
        }


embedding_service = EmbeddingService()
