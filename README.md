# MIRROR - AI-Powered Portfolio

Portfolio & knowledge base for **Michail Berjaoui**, Lead ML/LLM Engineer. Built with Flask, Qdrant, BGE-M3, and Phi-4.

## Features

- **Portfolio Landing** - Dark cyber-violet showcase of CV, skills, experience
- **AI Chat (RAG)** - Ask questions about uploaded documents with source citations
- **Web Scraper** - Input a URL, scrape it, ask questions about the content
- **Document Upload** - Drag & drop PDF, DOCX, TXT, MD files for indexing
- **Articles** - Markdown-based technical blog
- **Model Manager** - Load/unload/switch LLM models at runtime
- **Technical Choices** - Full documentation of architecture decisions

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask 3.1, Python 3.11+ |
| LLM | Phi-4 14B Q4_K_M via llama-cpp-python |
| Embedding | BGE-M3 via sentence-transformers |
| Vector DB | Qdrant (Docker, HNSW + INT8 quantization) |
| PDF Parser | PyMuPDF |
| Scraper | trafilatura + BeautifulSoup |

## Quick Start

### 1. Start Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant:v1.12.4
```

### 2. Download a GGUF model

Place a `.gguf` model in the `./models/` directory:

```bash
mkdir -p models
# Example: download Phi-4 Q4_K_M
# huggingface-cli download microsoft/phi-4-gguf phi-4-Q4_K_M.gguf --local-dir ./models/
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python run.py
```

Open [http://localhost:5000](http://localhost:5000)

### 5. Load models

Click the ⚙ button in the top-right corner to:
1. Connect to Qdrant
2. Load the embedding model (BGE-M3)
3. Load the LLM (select a .gguf file)

## Docker Compose

```bash
docker compose up -d
```

This starts both Qdrant and the Flask app.

## Infrastructure

Optimized for **64 GB RAM, 12 CPU cores, no GPU**:
- LLM: 10 threads, ~9 GB RAM
- Embedding: 2 threads, ~2 GB RAM
- Qdrant: 2 CPUs, ~4 GB RAM
- Remaining: OS + headroom

See [docs/choix_techniques.md](docs/choix_techniques.md) for full technical documentation.

## Project Structure

```
MIRROR/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── config.py            # All configuration dataclasses
│   ├── routes/              # Flask blueprints
│   │   ├── main.py          # Landing page
│   │   ├── chat.py          # RAG chat API
│   │   ├── documents.py     # Document upload/management
│   │   ├── scraper.py       # Web scraping API
│   │   ├── articles.py      # Articles API
│   │   └── models_route.py  # Model management API
│   ├── services/            # Business logic
│   │   ├── embedding.py     # BGE-M3 embedding service
│   │   ├── llm.py           # Phi-4 LLM service
│   │   ├── qdrant_store.py  # Qdrant vector store
│   │   ├── rag.py           # RAG pipeline with citations
│   │   ├── scraper.py       # Web scraping
│   │   └── pdf_parser.py    # Document parsing + chunking
│   ├── static/css/style.css # Dark cyber-violet theme
│   ├── static/js/app.js     # Global JS (model manager, toasts)
│   └── templates/           # Jinja2 templates
├── articles/                # Markdown articles
├── docs/choix_techniques.md # Technical choices document
├── models/                  # GGUF model files
├── uploads/                 # Uploaded documents
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── run.py
```

## License

Private portfolio project by Michail Berjaoui.
