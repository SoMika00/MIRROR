"""
Reranker service using cross-encoder/ms-marco-MiniLM-L-6-v2.

Chosen for:
  - Only 22M parameters → extremely fast on CPU (~5-15ms per query-doc pair)
  - Trained on MS MARCO passage ranking dataset (500M+ pairs)
  - NDCG@10 of 39.01 on TREC DL 2019 — best speed/quality for CPU
  - Compatible with sentence-transformers CrossEncoder API
  - MIT license

The reranker re-scores the top-K results from vector search using
cross-attention (joint query+document encoding), which is more accurate
than cosine similarity but too slow for first-stage retrieval.
"""

import time
import logging
import threading
from typing import List, Tuple

from app.config import reranker_cfg

logger = logging.getLogger(__name__)


class RerankerService:
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
        self.model_name = reranker_cfg.model_name
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        logger.info(f"Loading reranker: {self.model_name}")
        start = time.time()
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(self.model_name, device=reranker_cfg.device)
        elapsed = time.time() - start
        logger.info(f"Reranker loaded in {elapsed:.1f}s")
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def rerank(self, query: str, texts: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Rerank texts by relevance to query.
        Returns list of (original_index, score) sorted by score descending.
        """
        if not self._loaded:
            self.load()

        pairs = [(query, text) for text in texts]
        start = time.time()
        scores = self.model.predict(pairs)
        elapsed = time.time() - start

        # Build (index, score) and sort by score descending
        indexed_scores = [(i, float(s)) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Reranked {len(texts)} docs in {elapsed*1000:.1f}ms")
        return indexed_scores[:top_k]

    def get_info(self) -> dict:
        return {
            "model": self.model_name,
            "loaded": self._loaded,
            "device": reranker_cfg.device,
        }


reranker_service = RerankerService()
