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

SYSTEM_PROMPT = """You are MIRROR, an AI assistant powering Michail Berjaoui's portfolio website.
You have access to retrieved document excerpts below AND personal context about Michail.

""" + PERSONAL_CONTEXT + """

INSTRUCTIONS:
1. Read ALL the document excerpts carefully before answering.
2. Synthesize information across multiple sources when relevant.
3. Cite every claim from documents using [Source: name, p.X] format.
4. For personal facts (ABOUT section), answer directly without citation.
5. Answer in the SAME language as the question (French, English, or Japanese).
6. If the documents contain relevant code, include it in your answer.
7. Be precise, structured, and professional. Use bullet points or numbered lists when helpful.
8. If the retrieved documents don't answer the question, say so explicitly — do NOT hallucinate.

RETRIEVED DOCUMENTS (ordered by relevance):
{context}

---
USER QUESTION: {question}

ANSWER:"""

DIRECT_CHAT_PROMPT = """You are MIRROR, an AI assistant powering Michail Berjaoui's portfolio website.

""" + PERSONAL_CONTEXT + """

INSTRUCTIONS:
- Answer in the SAME language as the question (French, English, or Japanese).
- Be concise, helpful, and professional.
- For personal facts, answer directly and confidently.
- If asked about something you don't know, say so honestly.
- You can have natural conversations — greetings, technical discussions, etc.
- When discussing technical topics (AI, RAG, LLM, MLOps), provide detailed, expert-level answers.

CONVERSATION:
{history}USER: {question}

ASSISTANT:"""

SCRAPER_PROMPT = """You are MIRROR, an AI assistant. Answer questions based ONLY on the following 
web page content. Cite specific parts of the page.

PAGE: {title}
URL: {url}
CONTENT:
{context}

QUESTION: {question}

ANSWER (with citations from the page):"""


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


def query_rag(question: str, source_type: Optional[str] = None,
              stream: bool = False, enabled_sources: Optional[List[str]] = None) -> Dict[str, Any]:
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
                     enabled_sources: Optional[List[str]] = None) -> Generator:
    """Streaming RAG: yields answer tokens (str) and a sources dict."""
    query_vector = embedding_service.encode_query(question).tolist()
    results = qdrant_store.search(
        query_vector=query_vector,
        top_k=rag_cfg.top_k,
        score_threshold=rag_cfg.score_threshold,
        source_type=source_type,
        source_names=enabled_sources,
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
    prompt = SYSTEM_PROMPT.format(context=context, question=question)

    for token in llm_service.generate_stream(prompt):
        yield token


def query_direct_chat(question: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Direct chat without RAG — just the LLM + personal context."""
    start = time.time()

    history_text = ""
    if history:
        for msg in history[-6:]:
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            history_text += f"{role}: {msg['content']}\n"

    prompt = DIRECT_CHAT_PROMPT.format(question=question, history=history_text)

    t0 = time.time()
    answer = llm_service.generate(prompt)
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
    history_text = ""
    if history:
        for msg in history[-6:]:
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            history_text += f"{role}: {msg['content']}\n"

    prompt = DIRECT_CHAT_PROMPT.format(question=question, history=history_text)
    for token in llm_service.generate_stream(prompt):
        yield token


def query_scraped_content(question: str, url: str, title: str, content: str) -> Dict[str, Any]:
    """RAG over scraped web page content (no vector store needed)."""
    start = time.time()

    # Dynamic context limit based on current model's n_ctx
    n_ctx = getattr(llm_service, '_current_n_ctx', llm_cfg.n_ctx)
    max_context_chars = max(1000, (n_ctx - llm_cfg.max_tokens - 400) * 3)
    prompt = SCRAPER_PROMPT.format(
        title=title,
        url=url,
        context=content[:max_context_chars],
        question=question,
    )

    answer = llm_service.generate(prompt)
    total_time = time.time() - start

    return {
        "answer": answer,
        "sources": [{"source": url, "source_type": "web", "title": title}],
        "timings": {"total_ms": round(total_time * 1000, 1)},
    }
