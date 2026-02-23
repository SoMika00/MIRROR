"""
RAG (Retrieval-Augmented Generation) pipeline with source citations.

Architecture:
  1. Query → BGE-M3 embedding
  2. Qdrant vector search (top-k with score threshold)
  3. Context assembly with source tracking
  4. Phi-4 generation with citation-aware prompt
  5. Response with inline [source] citations

Based on:
  - Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
  - Qdrant hybrid search best practices
  - Microsoft Phi-4 technical report
"""

import logging
import time
from typing import List, Dict, Any, Optional, Generator

from app.services.embedding import embedding_service
from app.services.llm import llm_service
from app.services.reranker import reranker_service
from app.services.qdrant_store import qdrant_store, SearchResult
from app.config import rag_cfg, reranker_cfg

logger = logging.getLogger(__name__)

PERSONAL_CONTEXT = """ABOUT MICHAIL BERJAOUI (always available, no citation needed):
- Lead AI/LLM Engineer, 5 years experience deploying AI models to production at scale
- Enterprise RAG specialist, fine-tuning (LoRA/QLoRA), OCR/NER
- Currently based in Tokyo, Japan — actively seeking opportunities in Japan
- Passionate about understanding the agents that compose our world, thrives in event modeling
- Practices sport regularly, values discipline and physical well-being
- French (native), English (B2 professional), learning Japanese
- Education: Master MIASHS (Applied Mathematics & CS) — Université Paul Valéry Montpellier (2019-2021), Licence Maths Appliquées — Université Nice Sophia Antipolis (2015-2018)
- Certifications: Google Cloud — How Google does Machine Learning, Modernizing Data Lakes & Data Warehouses
- Top skills: Python, LLM, RAG, MLOps, SQL
- 500+ LinkedIn connections, 516 followers
- Open to: Data Science, Data Engineer, Chief Data Architect, AI Engineer, ML Engineer roles
- Interests: Anthropic, OpenAI, United World Inc, Noeon Research"""

SYSTEM_PROMPT = """You are MIRROR, an AI assistant for Michail Berjaoui's portfolio website.
You answer questions using the provided context documents AND the personal context below.
If the documents don't contain enough information, use personal context. If neither helps, say so.

""" + PERSONAL_CONTEXT + """

RULES:
- Answer in the same language as the question (French, English, or Japanese)
- Cite your sources using [Source: name, page X] format after each claim from documents
- For personal facts (from ABOUT section above), no citation is needed
- Be concise and professional
- Never invent information not present in the context or personal info

CONTEXT DOCUMENTS:
{context}

QUESTION: {question}

ANSWER (with citations):"""

SCRAPER_PROMPT = """You are MIRROR, an AI assistant. Answer questions based ONLY on the following 
web page content. Cite specific parts of the page.

PAGE: {title}
URL: {url}
CONTENT:
{context}

QUESTION: {question}

ANSWER (with citations from the page):"""


def build_context(results: List[SearchResult]) -> str:
    """Build context string from search results with source attribution."""
    context_parts = []
    for i, r in enumerate(results):
        source_label = f"[{r.source}"
        if r.page:
            source_label += f", p.{r.page}"
        source_label += f"] (score: {r.score:.2f})"
        context_parts.append(f"--- Document {i+1} {source_label} ---\n{r.text}\n")
    return "\n".join(context_parts)


def query_rag(question: str, source_type: Optional[str] = None,
              stream: bool = False) -> Dict[str, Any]:
    """Execute full RAG pipeline: embed → search → generate."""
    start = time.time()

    # Step 1: Embed the question
    t0 = time.time()
    query_vector = embedding_service.encode_query(question).tolist()
    embed_time = time.time() - t0

    # Step 2: Vector search
    t0 = time.time()
    results = qdrant_store.search(
        query_vector=query_vector,
        top_k=rag_cfg.top_k,
        score_threshold=rag_cfg.score_threshold,
        source_type=source_type,
    )
    search_time = time.time() - t0

    if not results:
        return {
            "answer": "I don't have enough context to answer this question. Please upload relevant documents or try a different query.",
            "sources": [],
            "timings": {"total": time.time() - start},
        }

    # Step 3: Rerank
    rerank_time = 0
    if rag_cfg.rerank and reranker_service.is_loaded():
        t0 = time.time()
        texts = [r.text for r in results]
        reranked = reranker_service.rerank(question, texts, top_k=reranker_cfg.top_k)
        results = [results[idx] for idx, _ in reranked]
        rerank_time = time.time() - t0

    # Step 4: Build context
    context = build_context(results)

    # Step 5: Generate answer
    prompt = SYSTEM_PROMPT.format(context=context, question=question)

    t0 = time.time()
    answer = llm_service.generate(prompt)
    gen_time = time.time() - t0

    total_time = time.time() - start

    # Step 5: Build source citations
    sources = []
    seen = set()
    for r in results:
        key = (r.source, r.page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": r.source,
                "source_type": r.source_type,
                "page": r.page,
                "score": round(r.score, 3),
                "excerpt": r.text[:200] + "..." if len(r.text) > 200 else r.text,
            })

    return {
        "answer": answer,
        "sources": sources,
        "timings": {
            "embedding_ms": round(embed_time * 1000, 1),
            "search_ms": round(search_time * 1000, 1),
            "rerank_ms": round(rerank_time * 1000, 1),
            "generation_ms": round(gen_time * 1000, 1),
            "total_ms": round(total_time * 1000, 1),
        },
    }


def query_rag_stream(question: str, source_type: Optional[str] = None) -> Generator[str, None, None]:
    """Streaming RAG: yields answer tokens one by one."""
    query_vector = embedding_service.encode_query(question).tolist()
    results = qdrant_store.search(
        query_vector=query_vector,
        top_k=rag_cfg.top_k,
        score_threshold=rag_cfg.score_threshold,
        source_type=source_type,
    )

    if not results:
        yield "I don't have enough context to answer this question."
        return

    # Rerank
    if rag_cfg.rerank and reranker_service.is_loaded():
        texts = [r.text for r in results]
        reranked = reranker_service.rerank(question, texts, top_k=reranker_cfg.top_k)
        results = [results[idx] for idx, _ in reranked]

    context = build_context(results)
    prompt = SYSTEM_PROMPT.format(context=context, question=question)

    for token in llm_service.generate_stream(prompt):
        yield token


def query_scraped_content(question: str, url: str, title: str, content: str) -> Dict[str, Any]:
    """RAG over scraped web page content (no vector store needed)."""
    start = time.time()

    prompt = SCRAPER_PROMPT.format(
        title=title,
        url=url,
        context=content[:3000],  # Limit context to fit in n_ctx
        question=question,
    )

    answer = llm_service.generate(prompt)
    total_time = time.time() - start

    return {
        "answer": answer,
        "sources": [{"source": url, "source_type": "web", "title": title}],
        "timings": {"total_ms": round(total_time * 1000, 1)},
    }
