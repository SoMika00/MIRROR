---
title: "RAG in Production: Building a Complete System on a Single Server"
date: 2025-01-31
tags: RAG, Architecture, Vector DB, Embeddings, Production, Qdrant, BGE-M3, Reranking, Infrastructure
summary: "Lessons learned from building an end-to-end RAG pipeline for MIRROR - architecture, technical choices, bare-metal infrastructure, TLS, reverse proxy, and the patterns that survived production on a single CPU server."
---

# RAG in Production: Building a Complete System on a Single Server

## The Context

MIRROR is a complete RAG system running on a dedicated Hetzner server (12 vCPU, 64 GB RAM, zero GPU). Users upload PDFs, URLs, Markdown, and the system answers by citing its sources in French, English, or Japanese.

This article covers two things: the RAG pipeline itself (what worked, what didn't) and the infrastructure to run it in production (server, TLS, reverse proxy, monitoring). This is the guide I wish I had before starting.

## 1. The Architecture That Survived

After several iterations, here is the final pipeline:

```
User question
    |
    v
[Query Router]  <-- lightweight classification (regex + heuristics, <1ms)
    |
    |-- SIMPLE --> Direct LLM (no RAG, ~100ms)
    |
    +-- MEDIUM/COMPLEX --> Full RAG Pipeline:
            |
            v
        [BGE-M3 Embedding]    -->  1024 dimensions, ~30ms
            |
            v
        [Qdrant Search]       -->  top-12, threshold 0.35, HNSW, ~15ms
            |
            v
        [Cross-Encoder Rerank] --> ms-marco-MiniLM-L-6-v2, threshold -3.0, ~80ms
            |
            v
        [Context Assembly]     --> top-5, relevance scores, token budget
            |
            v
        [Llama 3.1 8B Q4_K_M] --> 8192 ctx, mlock, ~10-18 t/s
            |
            v
        Response with separate sources
```

Each component was chosen after testing alternatives.

## 2. The Embedding: Why BGE-M3

I tested `E5-large-v2`, `GTE-large`, `all-MiniLM-L6-v2`, and `BGE-M3`. Main criterion: multilingual quality (French + English + Japanese in the same index).

`all-MiniLM-L6-v2` is fast but its French embeddings are mediocre: recall@10 dropped to 0.6 in my tests, versus 0.85 in English. `E5-large-v2` is decent in multi-language but its 1024-dim vectors consumed too much RAM for the same quality as BGE-M3.

BGE-M3 won for three reasons:
1. Natively trained on 100+ languages, same performance in French, English, Japanese
2. Supports long passages (up to 8192 tokens), crucial for 768-token chunks
3. Dense + sparse in the same model (I only use dense, but the option is there)

The catch: BGE-M3 takes ~2.5 GB of RAM. On a 64 GB server where the LLM takes 3-9 GB and Qdrant 2-4 GB, every GB counts. You must precisely measure the consumption of each component.

## 3. Chunking: The 768 Tokens That Changed Everything

My first chunking was 512 tokens with 64 overlap. That's what tutorials recommend. It was bad.

The problem: my technical documents have dense paragraphs. At 512 tokens, you cut in the middle of an explanation. The LLM receives a sentence fragment, doesn't understand the context, and hallucinates to fill the gap.

I systematically tested 256, 512, 768, and 1024 tokens on a set of 50 manual question-answer pairs:

| Chunk size | Recall@5 | Faithfulness | Hallucination rate |
|-----------|----------|-------------|-------------------|
| 256       | 0.72     | 0.68        | 23%               |
| 512       | 0.78     | 0.74        | 18%               |
| **768**   | **0.83** | **0.81**    | **11%**           |
| 1024      | 0.80     | 0.79        | 13%               |

768 tokens was the sweet spot: long enough to preserve context, short enough for retrieval to remain precise. The 128-token overlap (17%) ensures no important sentence falls between two chunks.

Lesson: don't trust default values. Test on your data with your questions.

## 4. The Reranker: The Underestimated Secret Weapon

Without a reranker, the pipeline returned "close but not relevant" results. Cosine similarity is good for recall, bad for precision.

Concrete example: for the question "What is MIRROR's architecture?", dense retrieval returned:
1. A chunk about MIRROR's architecture (relevant)
2. A chunk about Transformer architecture (not relevant: high cosine sim because "architecture" is a dominant term)
3. A chunk about MIRROR's infrastructure choices (relevant)

The cross-encoder `ms-marco-MiniLM-L-6-v2` solves this in ~80ms. It takes the question + each chunk and produces a contextual relevance score, not just a vector distance.

My setup: retrieve top-12 (wide net), then the reranker keeps the top-5 with a threshold of -3.0 (anything below is noise). Moving from top-3 to top-5 was made possible by increasing the context window to 8192 tokens.

The reranker catch: it loads a ~80 MB BERT model in RAM and adds ~80ms of latency. On simple queries (greetings, personal questions), it's wasteful. Hence the query router.

## 5. The Query Router: Don't Treat Everything the Same

The query router classifies requests into three tiers without any LLM call, just regex and heuristics:

- **SIMPLE** (greetings, personal questions): direct LLM, no RAG, ~100ms
- **MEDIUM** (factual questions): standard RAG, ~2-5s
- **COMPLEX** (comparisons, multi-part, deep technical): extended RAG with more chunks, ~5-15s

Routing takes <1ms. On a realistic query mix, this reduces average latency by ~40% because the majority of interactions are greetings or simple questions.

The code is ~150 lines of Python. No ML, no trained classifier. Keyword lists, regex, and conditional logic. Sometimes the simplest solution is the best.

## 6. The Prompt: The Invisible Iteration

The RAG prompt went through about ten versions. Each instruction fixes an observed bug:

- Without "Read ALL documents", the model read the first chunk and ignored the others
- Without "Answer in the SAME language", it answered in English to a French question
- Without "do NOT hallucinate", it invented plausible but false answers
- Without instructions about sources, it paraphrased without citing, making verification impossible

A good RAG prompt is a requirements document. Each line fixes a bug observed in production.

## 7. Context Assembly

Assembling context for the LLM is not just "concatenating chunks":

- **Relevance tags**: each chunk is annotated `[Relevance: HIGH/MEDIUM/LOW]` with the numerical score. The LLM can weigh accordingly.
- **Dynamic token budget**: context is truncated to respect `n_ctx - max_tokens - prompt_overhead`. No blind truncation - we cut at the last complete chunk.
- **Relevance ordering**: most relevant chunks first. LLMs have a position bias (primacy effect), put the important stuff at the beginning.

## 8. Infrastructure: From Zero to Production

This is the part nobody documents. Here's how I set up MIRROR from A to Z on a single server.

### 8.1 The Server

Dedicated Hetzner AX102: 12 cores, 64 GB DDR4 RAM, 2x 512 GB NVMe. Cost: ~65 euros/month. No GPU. The choice is deliberate: prove that a complete RAG can run on modest hardware.

Why a dedicated server and not cloud (AWS/GCP)? Because for a personal project with LLM inference running continuously, dedicated is 3-5x cheaper than equivalent cloud instances. No auto-scaling needed, no network latency between services.

### 8.2 Domain Name and DNS

DNS configuration:
- A record pointing to the server IP
- AAAA record if IPv6 is available
- 300s TTL to enable fast migration

### 8.3 TLS and Reverse Proxy with Caddy

Caddy is the ideal choice for a solo server:

```
mydomain.com {
    reverse_proxy /api/* localhost:5000 {
        flush_interval -1
    }
    reverse_proxy localhost:5000 {
        flush_interval -1
    }
}
```

Caddy automatically handles:
- TLS certificate via Let's Encrypt (zero configuration)
- Automatic renewal every 60 days
- HTTP/2 and HTTP/3 by default
- HTTP to HTTPS redirect

The `flush_interval -1` is critical for SSE (Server-Sent Events) streaming. Without it, Caddy buffers responses and streaming doesn't work.

### 8.4 Docker Compose: Orchestrating Everything

Three services in a single `docker-compose.yml`:

```yaml
services:
  caddy:        # Reverse proxy + TLS
  qdrant:       # Vector DB
  flask-app:    # Application (LLM + embedding + reranker)
```

Critical points:
- `mlock: true` in llama.cpp options to prevent model swapping
- Persistent volume for Qdrant (embeddings survive restarts)
- Volume for Caddy certificates
- Health checks on each service
- Restart policy `unless-stopped`

### 8.5 Production Monitoring

Metrics exposed via `/api/models/metrics`:
- **embed_time**: question embedding time (~30ms)
- **search_time**: Qdrant search time (~15ms)
- **rerank_time**: cross-encoder time (~80ms)
- **generation_time**: LLM generation time (2-15s depending on complexity)
- **tokens/sec**: real-time generation speed
- Per-core CPU, RAM, process RSS, live display in the interface

## 9. What Did NOT Work

### Semantic Query Cache
Caching responses by embedding similarity. In theory, massive latency gains. In practice, with conversation history changing the context, hits were rare and false positives dangerous.

### Semantic Chunking via Embeddings
Splitting when cosine similarity between adjacent windows drops. Too slow at ingestion, results not significantly better than fixed chunking at 768 tokens.

### Query Expansion with LLM
Reformulating the question with an LLM call before retrieval. +5% recall, but +2-3s latency. On CPU, every second counts. The reranker provides a better gain for less cost.

## Conclusion

A production RAG is not a linear pipeline you plug in and it works. It's a system with dozens of interdependent parameters that must be calibrated on your own data. And it's also a complete infrastructure stack: server, DNS, TLS, reverse proxy, containers, monitoring.

Three things I wish I had known before starting:
1. **Chunking is the most impactful parameter**: more than the embedding model, more than the LLM
2. **The reranker is worth every millisecond**: it's the best quality/cost ratio in the pipeline
3. **Infrastructure is 50% of the work**: a good RAG on bad infrastructure is useless

*Michail Berjaoui - January 2025*
