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
from typing import List, Dict, Any, Optional, Generator, Tuple

from app.services.embedding import embedding_service
from app.services.llm import llm_service
from app.services.reranker import reranker_service
from app.services.qdrant_store import qdrant_store, SearchResult
from app.config import rag_cfg, reranker_cfg, llm_cfg

logger = logging.getLogger(__name__)

PERSONAL_CONTEXT = """ABOUT MICHAIL BERJAOUI (always available, no citation needed):
- Lead AI/LLM Engineer, 5 years experience deploying AI models to production at scale
- Enterprise RAG specialist, fine-tuning (LoRA/QLoRA), OCR/NER
- Currently based in Tokyo, Japan - actively seeking opportunities in Japan
- Passionate about understanding the agents that compose our world, thrives in event modeling
- Deeply curious, loves learning - especially in technology. Not afraid to stay up late to solve a hard problem.
- Practices sport regularly, values discipline and physical well-being. Generally in a good mood.
- Values collaboration: enjoys working with teammates, learning from them, and sharing knowledge. Believes the best ideas come from the team.
- Outside work: appreciates quality time with friends, good sleep, and staying active
- French (native), English (B2 professional), learning Japanese
- Education: Master MIASHS (Applied Mathematics & CS) - Université Paul Valéry Montpellier (2019-2021), Licence Maths Appliquées - Université Nice Sophia Antipolis (2015-2018)
- Certifications: Google Cloud - How Google does Machine Learning, Modernizing Data Lakes & Data Warehouses
- Top skills: Python, LLM, RAG, MLOps, SQL
- 500+ LinkedIn connections, 516 followers
- Open to: Data Science, Data Engineer, Chief Data Architect, AI Engineer, ML Engineer roles
- Interests: Anthropic, OpenAI, United World Inc, Noeon Research
NOTE: Present these traits naturally when relevant. Do not exaggerate or over-praise. Keep a professional, grounded tone."""

RAG_SYSTEM = """You are MIRROR, Michail Berjaoui's AI assistant on his portfolio website.

""" + PERSONAL_CONTEXT + """

RULES:
- Read ALL retrieved documents before answering.
- Cite claims from documents: [Source: name, p.X].
- For personal facts (ABOUT section), answer directly.
- Answer in the SAME language as the question.
- Be concise, structured, professional.
- If documents don't answer the question, say so. Do NOT hallucinate."""

DIRECT_CHAT_SYSTEM = """You are MIRROR, Michail Berjaoui's AI assistant on his portfolio website.

""" + PERSONAL_CONTEXT + """

RULES:
- Answer in the SAME language as the question (French, English, or Japanese).
- Be concise, helpful, and professional.
- For greetings, respond naturally and briefly.
- For personal facts about Michail, answer directly and confidently.
- For technical topics (AI, RAG, LLM, MLOps), give detailed answers.
- If you don't know something, say so honestly. Never invent facts.
- Keep responses focused. Do NOT repeat the user's message back."""

SCRAPER_SYSTEM = """You are MIRROR, Michail Berjaoui's AI assistant on his portfolio website.

RULES:
- Answer questions based ONLY on the provided web page content.
- Cite specific parts of the page.
- Answer in the SAME language as the question.
- Be concise and structured. Do NOT hallucinate beyond the page content."""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English, ~2 for CJK."""
    return len(text) // 3


def build_context(results: List[SearchResult], max_tokens: int = 0) -> str:
    """Build context string from search results with source attribution.
    Truncates to fit within max_tokens budget (0 = no limit)."""
    if max_tokens <= 0:
        # Reserve tokens for system prompt (~800) + generation (max_tokens config)
        n_ctx = getattr(llm_service, '_current_n_ctx', llm_cfg.n_ctx)
        max_tokens = max(1024, n_ctx - llm_cfg.max_tokens - 1000)

    context_parts = []
    total_tokens = 0
    for i, r in enumerate(results):
        source_label = f"{r.source}"
        if r.page:
            source_label += f", p.{r.page}"
        relevance = "HIGH" if r.score > 0.7 else "MEDIUM" if r.score > 0.5 else "LOW"
        chunk = f"[Document {i+1}] Source: {source_label} | Relevance: {relevance} ({r.score:.2f})\n{r.text}\n"
        chunk_tokens = _estimate_tokens(chunk)
        if total_tokens + chunk_tokens > max_tokens:
            # Truncate this chunk to fill remaining budget
            remaining_chars = (max_tokens - total_tokens) * 3
            if remaining_chars > 100:
                context_parts.append(chunk[:remaining_chars] + "...")
            break
        context_parts.append(chunk)
        total_tokens += chunk_tokens

    if not context_parts:
        return "(No relevant documents found)"
    return "\n".join(context_parts)


def _build_rag_messages(question: str, context: str) -> list:
    """Build structured chat messages for RAG generation."""
    return [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": f"RETRIEVED DOCUMENTS:\n{context}\n\nQUESTION: {question}"},
    ]


def _build_chat_messages(question: str, history: Optional[List[Dict]] = None) -> list:
    """Build structured chat messages for direct chat."""
    messages = [{"role": "system", "content": DIRECT_CHAT_SYSTEM}]
    if history:
        for msg in history[-6:]:
            role = "user" if msg["role"] == "user" else "assistant"
            # Truncate long history messages to avoid overwhelming small models
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def query_rag(question: str, source_type: Optional[str] = None,
              stream: bool = False, enabled_sources: Optional[List[str]] = None,
              user_id: Optional[str] = None) -> Dict[str, Any]:
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
        source_names=enabled_sources,
        user_id=user_id,
    )
    search_time = time.time() - t0

    if not results:
        return {
            "answer": "I don't have enough context to answer this question. Please upload relevant documents or try a different query.",
            "sources": [],
            "timings": {"total": time.time() - start},
        }

    # Step 3: Rerank + score-based filtering
    rerank_time = 0
    if rag_cfg.rerank and reranker_service.is_loaded():
        t0 = time.time()
        texts = [r.text for r in results]
        reranked = reranker_service.rerank(question, texts, top_k=reranker_cfg.top_k)
        # Filter out clearly irrelevant results (cross-encoder score < -3)
        reranked = [(idx, score) for idx, score in reranked if score > -3.0]
        if reranked:
            results = [results[idx] for idx, _ in reranked]
            # Update scores with reranker scores for better context ordering
            for i, (_, rerank_score) in enumerate(reranked[:len(results)]):
                results[i] = SearchResult(
                    text=results[i].text,
                    score=max(results[i].score, (rerank_score + 10) / 20),  # normalize to ~0-1
                    source=results[i].source,
                    source_type=results[i].source_type,
                    page=results[i].page,
                    chunk_index=results[i].chunk_index,
                    metadata=results[i].metadata,
                )
        rerank_time = time.time() - t0
        logger.info(f"Reranked to {len(results)} results in {rerank_time*1000:.0f}ms")

    # Step 4: Build context
    context = build_context(results)

    # Step 5: Generate answer with structured messages
    messages = _build_rag_messages(question, context)

    t0 = time.time()
    answer = llm_service.generate(messages=messages)
    gen_time = time.time() - t0

    total_time = time.time() - start

    # Step 5: Build source citations
    sources = []
    seen = set()
    for r in results:
        key = (r.source, r.page, r.chunk_index)
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": r.source,
                "source_type": r.source_type,
                "page": r.page,
                "chunk_index": r.chunk_index,
                "score": round(r.score, 3),
                "excerpt": r.text[:200] + "..." if len(r.text) > 200 else r.text,
            })

    result = {
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

    return result


def query_rag_stream(question: str, source_type: Optional[str] = None,
                     enabled_sources: Optional[List[str]] = None,
                     user_id: Optional[str] = None) -> Generator:
    """Streaming RAG: yields answer tokens (str) and a sources dict."""
    query_vector = embedding_service.encode_query(question).tolist()
    results = qdrant_store.search(
        query_vector=query_vector,
        top_k=rag_cfg.top_k,
        score_threshold=rag_cfg.score_threshold,
        source_type=source_type,
        source_names=enabled_sources,
        user_id=user_id,
    )

    if not results:
        yield "I don't have enough context to answer this question."
        return

    # Rerank + score filter
    if rag_cfg.rerank and reranker_service.is_loaded():
        texts = [r.text for r in results]
        reranked = reranker_service.rerank(question, texts, top_k=reranker_cfg.top_k)
        reranked = [(idx, score) for idx, score in reranked if score > -3.0]
        if reranked:
            results = [results[idx] for idx, _ in reranked]

    # Emit sources before tokens so the frontend knows what was retrieved
    sources = []
    seen = set()
    for r in results:
        key = (r.source, r.page, r.chunk_index)
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": r.source,
                "source_type": r.source_type,
                "page": r.page,
                "chunk_index": r.chunk_index,
                "score": round(r.score, 3),
            })
    yield {"sources": sources}

    context = build_context(results)
    messages = _build_rag_messages(question, context)

    for token in llm_service.generate_stream(messages=messages):
        yield token


def query_direct_chat(question: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Direct chat without RAG - just the LLM + personal context."""
    start = time.time()

    messages = _build_chat_messages(question, history)

    t0 = time.time()
    answer = llm_service.generate(messages=messages)
    gen_time = time.time() - t0
    total_time = time.time() - start

    return {
        "answer": answer,
        "sources": [],
        "timings": {
            "generation_ms": round(gen_time * 1000, 1),
            "total_ms": round(total_time * 1000, 1),
        },
    }


def query_direct_chat_stream(question: str, history: Optional[List[Dict]] = None) -> Generator[str, None, None]:
    """Streaming direct chat."""
    messages = _build_chat_messages(question, history)
    for token in llm_service.generate_stream(messages=messages):
        yield token


def query_scraped_content(question: str, url: str, title: str, content: str) -> Dict[str, Any]:
    """RAG over scraped web page content (no vector store needed)."""
    start = time.time()

    # Dynamic context limit based on current model's n_ctx
    n_ctx = getattr(llm_service, '_current_n_ctx', llm_cfg.n_ctx)
    max_context_chars = max(1000, (n_ctx - llm_cfg.max_tokens - 500) * 3)
    truncated = content[:max_context_chars]

    messages = [
        {"role": "system", "content": SCRAPER_SYSTEM},
        {"role": "user", "content": f"PAGE: {title}\nURL: {url}\n\nCONTENT:\n{truncated}\n\nQUESTION: {question}"},
    ]

    answer = llm_service.generate(messages=messages)
    total_time = time.time() - start

    return {
        "answer": answer,
        "sources": [{"source": url, "source_type": "web", "title": title}],
        "timings": {"total_ms": round(total_time * 1000, 1)},
    }
