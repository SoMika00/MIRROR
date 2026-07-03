# MIRROR — AI-Powered Portfolio

Portfolio & live RAG application for **Michail Berjaoui**, Lead AI/LLM Engineer.
Fully **API-first**: no model weights run on the server.

**Live:** https://mymirror.fr

## Features

- **Portfolio landing** — CV, selected work, skills, experience
- **AI Chat (RAG)** — the assistant answers questions about Michail with
  citations; the site indexes its own content (profile, articles, architecture
  docs) at startup
- **Bring your own data** — visitors can upload PDF/DOCX/TXT/MD or scrape a
  URL and query it (per-user isolation via anonymous cookie sessions)
- **Articles** — markdown-based technical blog (FR/EN)
- **Architecture page** — every technical decision documented like a client
  deliverable, including the V1 (self-hosted) → V2 (API-first) migration story
- **Cost guardrail** — hard daily LLM budget (default $0.50/day) enforced
  before every API call

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask 3.1, Python 3.11, gunicorn |
| LLM | xAI Grok API (OpenAI-compatible SDK) |
| Retrieval | SQLite FTS5 (BM25), hybrid-ready with optional API embeddings |
| App state | SQLite (WAL) |
| Documents | PyMuPDF, python-docx |
| Scraper | trafilatura + BeautifulSoup |
| Edge | Caddy 2 (automatic HTTPS) |
| Deploy | Docker Compose |

## Quick Start

```bash
cp .env.example .env   # add your GROK_API_KEY
docker compose up -d --build
```

Open http://localhost — that's it. No models to download, no vector DB to run.

### Local development (without Docker)

```bash
pip install -r requirements.txt
python run.py
```

## Configuration (.env)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROK_API_KEY` | — | xAI API key (required) |
| `GROK_MODEL` | `grok-4.20-non-reasoning` | Chat model |
| `GROK_DAILY_BUDGET` | `0.50` | Hard daily spend cap (USD) |
| `GROK_INPUT_PRICE` / `GROK_OUTPUT_PRICE` | `1.25` / `2.50` | USD per 1M tokens, for budget accounting |
| `EMBEDDINGS_API_KEY` | empty | Set to enable hybrid dense+BM25 retrieval via any OpenAI-compatible embeddings API |

## Deployment notes

- DNS: point the `A` record of your domain at the server's public IP.
  Caddy then obtains the TLS certificate automatically (no action needed).
- Publishing an article = drop a markdown file in `articles/`
  (`slug.md`, `slug.en.md`, `slug.ja.md`); it is indexed at next boot.

## Architecture decisions

See [docs/choix_techniques.md](docs/choix_techniques.md) (FR) or the
[/tech](https://mymirror.fr/tech) page (EN). Both are indexed into the
assistant's knowledge base — you can ask the AI chat why any decision was made.
