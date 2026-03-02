"""
Qdrant vector store service.

Qdrant chosen for:
  - Written in Rust → low memory footprint, high throughput
  - HNSW index with configurable m/ef parameters
  - Scalar INT8 quantization → 4× RAM reduction with <1% recall loss
  - Rich metadata filtering (source, page, type, timestamp)
  - Active open-source community (17k+ GitHub stars)
  - Self-hosted → full data sovereignty (important for Japan market)
  - gRPC + REST API

Configuration optimized for 64GB RAM / 12 CPU:
  - HNSW m=16, ef_construct=200 → good recall/build speed balance
  - ef_search=128 → sub-10ms search latency
  - INT8 scalar quantization with always_ram=True
  - Payload indexing on 'source_type' and 'source_name' for fast filtering
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.config import qdrant_cfg

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    score: float
    source: str
    source_type: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class QdrantStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.client = None
        self._connected = False

    def connect(self):
        if self._connected:
            return
        from qdrant_client import QdrantClient
        logger.info(f"Connecting to Qdrant at {qdrant_cfg.host}:{qdrant_cfg.port}")
        self.client = QdrantClient(host=qdrant_cfg.host, port=qdrant_cfg.port, timeout=10)
        self._connected = True
        self._ensure_collection()

    def is_connected(self) -> bool:
        return self._connected

    def _ensure_collection(self):
        from qdrant_client.models import (
            VectorParams, Distance, ScalarQuantization,
            ScalarQuantizationConfig, ScalarType,
            HnswConfigDiff, PayloadSchemaType,
        )
        collections = [c.name for c in self.client.get_collections().collections]
        if qdrant_cfg.collection_name not in collections:
            logger.info(f"Creating collection: {qdrant_cfg.collection_name}")
            self.client.create_collection(
                collection_name=qdrant_cfg.collection_name,
                vectors_config=VectorParams(
                    size=qdrant_cfg.vector_size,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=qdrant_cfg.hnsw_m,
                    ef_construct=qdrant_cfg.hnsw_ef_construct,
                ),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        always_ram=qdrant_cfg.always_ram,
                    ),
                ),
            )
            # Payload indexes for fast filtering
            self.client.create_payload_index(
                collection_name=qdrant_cfg.collection_name,
                field_name="source_type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=qdrant_cfg.collection_name,
                field_name="source_name",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=qdrant_cfg.collection_name,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Collection created with HNSW + INT8 quantization + payload indexes")
        else:
            logger.info(f"Collection '{qdrant_cfg.collection_name}' already exists")

    def upsert(self, texts: List[str], vectors: List[List[float]],
               payloads: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        if not self._connected:
            self.connect()
        import uuid
        from qdrant_client.models import PointStruct

        points = []
        for i, (text, vector, payload) in enumerate(zip(texts, vectors, payloads)):
            point_id = ids[i] if ids else str(uuid.uuid4())
            payload["text"] = text
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            ))

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=qdrant_cfg.collection_name,
                points=batch,
            )
        logger.info(f"Upserted {len(points)} vectors")

    def search(self, query_vector: List[float], top_k: int = 5,
               score_threshold: float = 0.0,
               source_type: Optional[str] = None,
               source_names: Optional[List[str]] = None,
               user_id: Optional[str] = None) -> List[SearchResult]:
        if not self._connected:
            self.connect()
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, SearchParams

        conditions = []
        if source_type:
            conditions.append(FieldCondition(key="source_type", match=MatchValue(value=source_type)))
        if source_names:
            conditions.append(FieldCondition(key="source_name", match=MatchAny(any=source_names)))
        if user_id:
            conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        query_filter = Filter(must=conditions) if conditions else None

        start = time.time()
        results = self.client.search(
            collection_name=qdrant_cfg.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            search_params=SearchParams(
                hnsw_ef=qdrant_cfg.hnsw_ef_search,
                exact=False,
            ),
        )
        elapsed = time.time() - start
        logger.debug(f"Search completed in {elapsed*1000:.1f}ms, {len(results)} results")

        return [
            SearchResult(
                text=r.payload.get("text", ""),
                score=r.score,
                source=r.payload.get("source_name", "unknown"),
                source_type=r.payload.get("source_type", "unknown"),
                page=r.payload.get("page"),
                chunk_index=r.payload.get("chunk_index"),
                metadata=r.payload,
            )
            for r in results
        ]

    def delete_by_source(self, source_name: str, user_id: Optional[str] = None):
        if not self._connected:
            self.connect()
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = [FieldCondition(key="source_name", match=MatchValue(value=source_name))]
        if user_id:
            conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        self.client.delete(
            collection_name=qdrant_cfg.collection_name,
            points_selector=Filter(must=conditions),
        )
        logger.info(f"Deleted vectors for source: {source_name}")

    def get_collection_info(self) -> dict:
        if not self._connected:
            return {"connected": False}
        info = self.client.get_collection(qdrant_cfg.collection_name)
        return {
            "connected": True,
            "collection": qdrant_cfg.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    def list_sources(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._connected:
            self.connect()
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        scroll_filter = None
        if user_id:
            scroll_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )

        results = self.client.scroll(
            collection_name=qdrant_cfg.collection_name,
            limit=1000,
            with_payload=["source_name", "source_type"],
            with_vectors=False,
            scroll_filter=scroll_filter,
        )
        sources = {}
        for point in results[0]:
            name = point.payload.get("source_name", "unknown")
            stype = point.payload.get("source_type", "unknown")
            if name not in sources:
                sources[name] = {"source_name": name, "source_type": stype, "chunks": 0}
            sources[name]["chunks"] += 1
        return list(sources.values())


qdrant_store = QdrantStore()
