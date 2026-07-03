"""
RAG pipeline with source citations - fully API-first.

Architecture:
  1. Query -> hybrid retrieval (SQLite FTS5 BM25, + API embeddings if enabled)
  2. Context assembly with source tracking
  3. Grok generation with a citation-aware prompt
  4. Response with [Source: ...] citations

The visitor-facing chat always searches the portfolio's own knowledge
(profile, articles, architecture docs) in addition to anything the visitor
uploaded or scraped.
"""

import logging
import time
from typing import Any, Dict, Generator, List, Optional

from app.services.llm import llm_service
from app.services.retrieval import retrieval_store, SearchResult
from app.config import rag_cfg

logger = logging.getLogger(__name__)

PERSONAL_CONTEXT = """ABOUT MICHAIL BERJAOUI (always available, no citation needed):
- Lead AI/LLM Engineer, 5+ years shipping AI systems to production at scale
- Enterprise RAG specialist, fine-tuning (LoRA/QLoRA), OCR/NER, MLOps
- Led teams of 4-8 engineers as Lead Data Scientist & AI Architect at SOMA
- Currently based in Tokyo, Japan - open to international teams (Japan) and French companies
- French (native), English (B2 professional), learning Japanese
- Education: Master MIASHS (Applied Mathematics & CS) - Université Paul Valéry Montpellier; Licence Maths Appliquées - Université Nice Sophia Antipolis
- Contact: michail.berjaoui@gmail.com · linkedin.com/in/mickail-berjaoui-02776b193
- This website (MIRROR) is itself his work: an API-first RAG chatbot (Grok API + SQLite FTS5 hybrid retrieval) - visitors can upload documents or scrape a URL and query them
NOTE: Present these facts naturally when relevant. Professional, grounded tone - no over-praise."""

RAG_SYSTEM = """You are MIRROR, the AI assistant on Michail Berjaoui's portfolio website. Your audience includes recruiters, product managers and AI experts evaluating Michail.

""" + PERSONAL_CONTEXT + """

RULES:
- Read ALL retrieved documents before answering.
- Cite claims taken from documents: [Source: name].
- For personal facts (ABOUT section), answer directly without citation.
- Answer in the SAME language as the question.
- Be concise, structured, professional.
- If the documents don't answer the question, say so plainly. Never invent facts."""

DIRECT_CHAT_SYSTEM = """You are MIRROR, the AI assistant on Michail Berjaoui's portfolio website. Your audience includes recruiters, product managers and AI experts evaluating Michail.

""" + PERSONAL_CONTEXT + """

RULES:
- Answer in the SAME language as the question (French, English, or Japanese).
- Be concise, helpful, and professional.
- For greetings, respond naturally and briefly.
- For personal facts about Michail, answer directly and confidently.
- For technical topics (AI, RAG, LLM, MLOps), give precise, expert-level answers.
- If you don't know something, say so honestly. Never invent facts.
- Keep responses focused. Do NOT repeat the user's message back."""

SCRAPER_SYSTEM = """You are MIRROR, the AI assistant on Michail Berjaoui's portfolio website.

RULES:
- Answer questions based ONLY on the provided web page content.
- Cite specific parts of the page.
- Answer in the SAME language as the question.
- Be concise and structured. Do NOT hallucinate beyond the page content."""

# Keep prompt+context well under the model's context and the daily budget.
MAX_CONTEXT_TOKENS = 6000


def _estimate_tokens(text: str) -> int:
    return len(text) // 3


def build_context(results: List[SearchResult], max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """Build context string from search results with source attribution."""
    context_parts = []
    total_tokens = 0
    for i, r in enumerate(results):
        source_label = f"{r.source}"
        if r.page:
            source_label += f", p.{r.page}"
        chunk = f"[Document {i+1}] Source: {source_label} (score {r.score:.2f})\n{r.text}\n"
        chunk_tokens = _estimate_tokens(chunk)
        if total_tokens + chunk_tokens > max_tokens:
            remaining_chars = (max_tokens - total_tokens) * 3
            if remaining_chars > 100:
                context_parts.append(chunk[:remaining_chars] + "...")
            break
        context_parts.append(chunk)
        total_tokens += chunk_tokens

    if not context_parts:
        return "(No relevant documents found)"
    return "\n".join(context_parts)


def _build_rag_messages(question: str, context: str, history: Optional[List[Dict]] = None) -> list:
    messages = [{"role": "system", "content": RAG_SYSTEM}]
    if history:
        for msg in history[-4:]:
            role = "user" if msg["role"] == "user" else "assistant"
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": f"RETRIEVED DOCUMENTS:\n{context}\n\nQUESTION: {question}"})
    return messages


def _build_chat_messages(question: str, history: Optional[List[Dict]] = None) -> list:
    messages = [{"role": "system", "content": DIRECT_CHAT_SYSTEM}]
    if history:
        for msg in history[-6:]:
            role = "user" if msg["role"] == "user" else "assistant"
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def _search(question: str, source_type: Optional[str], enabled_sources: Optional[List[str]],
            user_id: Optional[str]) -> List[SearchResult]:
    return retrieval_store.search(
        question=question,
        top_k=rag_cfg.top_k,
        source_type=source_type,
        source_names=enabled_sources,
        user_id=user_id,
        include_global=True,
    )


def _dedupe_sources(results: List[SearchResult]) -> List[Dict[str, Any]]:
    sources, seen = [], set()
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
    return sources


def query_rag(question: str, source_type: Optional[str] = None,
              enabled_sources: Optional[List[str]] = None,
              user_id: Optional[str] = None,
              history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Full RAG pipeline: search -> context -> generate."""
    start = time.time()

    t0 = time.time()
    results = _search(question, source_type, enabled_sources, user_id)
    search_time = time.time() - t0

    if not results:
        # Fall back to direct chat instead of a dead-end answer
        fallback = query_direct_chat(question, history=history)
        fallback["route_note"] = "no_retrieval_hits"
        return fallback

    context = build_context(results)
    messages = _build_rag_messages(question, context, history=history)

    t0 = time.time()
    answer = llm_service.generate(messages=messages)
    gen_time = time.time() - t0

    return {
        "answer": answer,
        "sources": _dedupe_sources(results),
        "timings": {
            "search_ms": round(search_time * 1000, 1),
            "generation_ms": round(gen_time * 1000, 1),
            "total_ms": round((time.time() - start) * 1000, 1),
        },
    }


def query_rag_stream(question: str, source_type: Optional[str] = None,
                     enabled_sources: Optional[List[str]] = None,
                     user_id: Optional[str] = None,
                     history: Optional[List[Dict]] = None) -> Generator:
    """Streaming RAG: yields a sources dict first, then answer tokens (str)."""
    results = _search(question, source_type, enabled_sources, user_id)

    if not results:
        for token in query_direct_chat_stream(question, history=history):
            yield token
        return

    yield {"sources": _dedupe_sources(results)}

    context = build_context(results)
    messages = _build_rag_messages(question, context, history=history)
    for token in llm_service.generate_stream(messages=messages):
        yield token


def query_direct_chat(question: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    start = time.time()
    messages = _build_chat_messages(question, history)
    answer = llm_service.generate(messages=messages)
    return {
        "answer": answer,
        "sources": [],
        "timings": {"total_ms": round((time.time() - start) * 1000, 1)},
    }


def query_direct_chat_stream(question: str, history: Optional[List[Dict]] = None) -> Generator[str, None, None]:
    messages = _build_chat_messages(question, history)
    for token in llm_service.generate_stream(messages=messages):
        yield token


def query_scraped_content(question: str, url: str, title: str, content: str) -> Dict[str, Any]:
    """Q&A over scraped web page content (page fits in context, no retrieval)."""
    start = time.time()
    truncated = content[:MAX_CONTEXT_TOKENS * 3]
    messages = [
        {"role": "system", "content": SCRAPER_SYSTEM},
        {"role": "user", "content": f"PAGE: {title}\nURL: {url}\n\nCONTENT:\n{truncated}\n\nQUESTION: {question}"},
    ]
    answer = llm_service.generate(messages=messages)
    return {
        "answer": answer,
        "sources": [{"source": url, "source_type": "web", "title": title}],
        "timings": {"total_ms": round((time.time() - start) * 1000, 1)},
    }
