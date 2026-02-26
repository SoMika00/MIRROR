"""
Semantic Query Cache — near-instant responses for similar questions.

Architecture:
  - Caches (query_embedding, answer, sources, timings) tuples
  - On new query: embed → cosine similarity against cache keys
  - If similarity > threshold (default 0.92): return cached answer instantly (<5ms)
  - LRU eviction when cache exceeds max_size
  - Thread-safe singleton

This is a production pattern used by companies like Notion and Perplexity
to reduce LLM costs and latency by 60-80% on repeated/similar queries.
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
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
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._embeddings: OrderedDict[str, np.ndarray] = OrderedDict()
        self._rw_lock = threading.Lock()
        self.max_size = int(__import__('os').environ.get("CACHE_MAX_SIZE", "200"))
        self.similarity_threshold = float(__import__('os').environ.get("CACHE_THRESHOLD", "0.92"))
        self._hits = 0
        self._misses = 0

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fast cosine similarity between two normalized vectors."""
        return float(np.dot(a, b))

    def lookup(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Check cache for a semantically similar query. Returns cached result or None."""
        if len(self._cache) == 0:
            self._misses += 1
            return None

        with self._rw_lock:
            best_score = 0.0
            best_key = None

            # Vectorized similarity against all cached embeddings
            if self._embeddings:
                keys = list(self._embeddings.keys())
                matrix = np.stack([self._embeddings[k] for k in keys])
                sims = matrix @ query_embedding  # assumes normalized vectors
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])
                best_key = keys[best_idx]

            if best_score >= self.similarity_threshold and best_key:
                self._hits += 1
                entry = self._cache[best_key]
                # Move to end (most recently used)
                self._cache.move_to_end(best_key)
                self._embeddings.move_to_end(best_key)
                logger.info(f"Cache HIT (sim={best_score:.3f}): {best_key[:60]}")
                return {
                    **entry,
                    "cache_hit": True,
                    "cache_similarity": round(best_score, 3),
                }

        self._misses += 1
        return None

    def store(self, query: str, query_embedding: np.ndarray, result: Dict[str, Any]):
        """Store a query result in the cache."""
        with self._rw_lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._embeddings.pop(oldest_key, None)

            self._cache[query] = {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "timings": result.get("timings", {}),
                "cached_at": time.time(),
            }
            self._embeddings[query] = query_embedding
            logger.debug(f"Cache STORE: {query[:60]} (size={len(self._cache)})")

    def invalidate(self, source_name: Optional[str] = None):
        """Clear cache entries. If source_name given, only clear entries citing that source."""
        with self._rw_lock:
            if source_name is None:
                self._cache.clear()
                self._embeddings.clear()
                logger.info("Cache fully invalidated")
            else:
                to_remove = []
                for key, entry in self._cache.items():
                    sources = entry.get("sources", [])
                    if any(s.get("source") == source_name for s in sources):
                        to_remove.append(key)
                for key in to_remove:
                    del self._cache[key]
                    self._embeddings.pop(key, None)
                if to_remove:
                    logger.info(f"Cache invalidated {len(to_remove)} entries for source: {source_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0.0,
            "threshold": self.similarity_threshold,
        }


semantic_cache = SemanticCache()
