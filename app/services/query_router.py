"""
Adaptive Query Router — automatic complexity classification and pipeline routing.

Architecture:
  - Classifies incoming queries into complexity tiers using lightweight heuristics
  - Routes to the optimal pipeline based on classification:
    * SIMPLE  → Direct chat (personal context only, no RAG overhead)
    * MEDIUM  → Standard RAG (embed → search → rerank → generate)
    * COMPLEX → Enhanced RAG (larger context window, more chunks, chain-of-thought prompt)
  - Zero-latency classification (<1ms) — no LLM call needed for routing
  - Logs routing decisions for observability

Complexity signals:
  - Query length (short = likely simple)
  - Question type (who/what/when = factual, how/why/compare = complex)
  - Technical keyword density
  - Multi-part detection (and/or/also/plus)
  - Personal vs document query detection
"""

import re
import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Personal topic keywords — route to direct chat
PERSONAL_KEYWORDS = {
    "michail", "berjaoui", "your", "you", "portfolio", "cv", "resume",
    "experience", "education", "skills", "contact", "linkedin", "email",
    "tokyo", "japan", "french", "english", "japanese", "hobby", "sport",
    "certification", "google cloud", "orange", "amadeus", "soma",
    "who are you", "tell me about", "introduce", "hello", "hi", "bonjour",
    "salut", "konnichiwa", "présente", "présentation", "parcours",
}

# Technical/document keywords — likely need RAG
TECHNICAL_KEYWORDS = {
    "rag", "llm", "embedding", "vector", "qdrant", "reranker", "inference",
    "quantization", "fine-tuning", "lora", "qlora", "transformer", "attention",
    "bert", "gpt", "phi", "qwen", "llama", "deployment", "kubernetes",
    "docker", "mlops", "pipeline", "architecture", "benchmark", "evaluation",
    "gguf", "vllm", "tensorrt", "ocr", "ner", "agent", "agentic",
    "openai", "google", "anthropic", "sdk", "mcp", "guardrail",
    "chunking", "retrieval", "generation", "prompt", "context window",
}

# Complex question patterns
COMPLEX_PATTERNS = [
    r"\b(compare|comparison|versus|vs\.?|difference|between)\b",
    r"\b(how does|how do|how would|how can|explain|describe)\b",
    r"\b(why|what are the advantages|what are the pros|trade-?off)\b",
    r"\b(step by step|walk me through|guide|tutorial)\b",
    r"\b(best practice|pattern|architecture|design)\b",
    r"\b(and also|in addition|furthermore|moreover|plus)\b",
    r"\b(benchmark|performance|optimize|improve)\b",
]

# Simple greeting/question patterns
SIMPLE_PATTERNS = [
    r"^(hi|hello|hey|bonjour|salut|yo|coucou)\b",
    r"^(who|what is your|what's your|tell me about yourself)",
    r"^(thanks|thank you|merci|ok|okay|got it)",
    r"^.{0,30}$",  # Very short queries
]


@dataclass
class RouteDecision:
    tier: str  # "simple", "medium", "complex"
    mode: str  # "chat", "rag"
    reason: str
    confidence: float  # 0.0-1.0
    params: Dict[str, Any]  # Override parameters for the chosen pipeline


def classify_query(question: str, has_sources: bool = False) -> RouteDecision:
    """
    Classify query complexity and determine optimal routing.
    Returns a RouteDecision with tier, mode, and parameter overrides.
    """
    q_lower = question.lower().strip()
    q_len = len(question)
    words = q_lower.split()
    word_count = len(words)

    # Score components
    personal_score = 0.0
    technical_score = 0.0
    complexity_score = 0.0

    # 1. Check personal keywords
    for kw in PERSONAL_KEYWORDS:
        if kw in q_lower:
            personal_score += 1.0
    personal_score = min(personal_score / 3.0, 1.0)

    # 2. Check technical keywords
    for kw in TECHNICAL_KEYWORDS:
        if kw in q_lower:
            technical_score += 1.0
    technical_score = min(technical_score / 3.0, 1.0)

    # 3. Check complexity patterns
    complex_matches = 0
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, q_lower):
            complex_matches += 1
    complexity_score = min(complex_matches / 2.0, 1.0)

    # 4. Check simple patterns
    is_simple_pattern = any(re.search(p, q_lower) for p in SIMPLE_PATTERNS)

    # 5. Length-based complexity
    if word_count > 20:
        complexity_score += 0.3
    elif word_count < 5:
        complexity_score -= 0.3

    # --- Routing Logic ---

    # Simple: greetings, personal questions without tech overlap, very short
    if is_simple_pattern and technical_score < 0.3 and not has_sources:
        return RouteDecision(
            tier="simple",
            mode="chat",
            reason="greeting/personal query",
            confidence=0.9,
            params={},
        )

    if personal_score > 0.5 and technical_score < 0.3:
        return RouteDecision(
            tier="simple",
            mode="chat",
            reason="personal question (no RAG needed)",
            confidence=0.8,
            params={},
        )

    # Complex: multi-part, comparison, deep technical, long
    if complexity_score > 0.6 or (technical_score > 0.5 and word_count > 15):
        top_k = 15  # More chunks for complex queries
        reranker_top_k = 5  # Keep more after reranking
        return RouteDecision(
            tier="complex",
            mode="rag" if has_sources else "chat",
            reason=f"complex query (complexity={complexity_score:.1f}, tech={technical_score:.1f})",
            confidence=0.7 + complexity_score * 0.2,
            params={
                "top_k": top_k,
                "reranker_top_k": reranker_top_k,
                "max_tokens": 1536,  # Allow longer answers
            },
        )

    # Medium: standard RAG or chat
    return RouteDecision(
        tier="medium",
        mode="rag" if (has_sources or technical_score > 0.3) else "chat",
        reason=f"standard query (tech={technical_score:.1f}, personal={personal_score:.1f})",
        confidence=0.7,
        params={},
    )
