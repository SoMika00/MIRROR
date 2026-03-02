# MIRROR - Choix Techniques

## Document de référence pour l'architecture du portfolio AI de Michail Berjaoui

---

## 1. Contraintes d'Infrastructure

| Ressource | Spécification | Allocation |
|-----------|--------------|------------|
| RAM | 64 Go DDR4/DDR5 | LLM ~12 Go · Embedding ~2 Go · Qdrant ~4 Go · OS+App ~6 Go · Libre ~40 Go |
| CPU | 12 cœurs (x86_64) | LLM: 10 threads · Embedding: 2 threads · Système: 2 threads |
| GPU | Aucun | Toutes les décisions de modèle sont orientées CPU |
| Stockage | SSD | Modèles ~10 Go · Données Qdrant · Documents uploadés |

**L'absence de GPU est la contrainte déterminante.** Chaque choix de modèle priorise la vitesse d'inférence CPU tout en maintenant une qualité production.

---

## 2. LLM : Phi-4 14B (Q4_K_M GGUF)

### Justification

- **Meilleur ratio qualité/taille à 14B** - Phi-4 de Microsoft égale ou dépasse des modèles 30B+ sur les benchmarks de raisonnement (MMLU 83.6, HumanEval 82.6, GSM8K 94.5)
- **Quantification Q4_K_M** - ~9 Go RAM avec dégradation de perplexité <0.5 vs FP16
- **Optimisé CPU via llama.cpp** - Format GGUF avec accélération SIMD AVX2/AVX-512, ~5-10 tokens/sec sur 12 cœurs
- **Licence MIT** - utilisation commerciale complète
- **Contexte 4096 tokens** - suffisant pour RAG avec 5 chunks de ~500 tokens

### Alternatives étudiées

| Modèle | Taille | RAM (Q4) | CPU t/s | Verdict |
|--------|--------|----------|---------|---------|
| **Phi-4 14B** | 14B | ~9 Go | 5-10 | **Sélectionné** |
| Mistral 7B | 7B | ~5 Go | 10-18 | Plus rapide mais qualité inférieure pour RAG complexe |
| Qwen2.5 14B | 14B | ~9 Go | 5-10 | Comparable, Phi-4 supérieur en raisonnement |
| Llama 3.3 70B | 70B | ~42 Go | 1-2 | Trop lent sur CPU, RAM trop juste |
| Phi-4-mini 3.8B | 3.8B | ~2.5 Go | 20-30 | Option fallback pour cas speed-critical |

### Moteur d'inférence : llama-cpp-python

Choisi plutôt qu'Ollama ou vLLM car :
- **Zéro overhead** - inférence C++ directe, pas de couche HTTP serveur
- **Contrôle fin** - n_threads, n_batch, n_ctx ajustables par requête
- **Hot-swap** - charger/décharger des modèles sans redémarrer Flask
- **Mémoire efficace** - support mmap

### Références

- Abdin et al. (2024). *"Phi-4 Technical Report"*. Microsoft Research. arXiv:2412.08905
- Dettmers et al. (2023). *"QLoRA: Efficient Finetuning of Quantized Language Models"*. NeurIPS 2023
- Benchmarks quantification GGUF llama.cpp - github.com/ggml-org/llama.cpp/discussions/3847

---

## 3. Embedding : BGE-M3

### Justification

- **Multilingue** - 100+ langues dont français, anglais et japonais (critique pour le marché japonais)
- **Retrieval hybride** - représentations dense (1024-dim) + sparse + multi-vecteur dans un seul modèle
- **Performance MTEB** - 63.0 MTEB score, meilleur modèle embedding open-source pour retrieval
- **CPU-friendly** - 567M paramètres, <30ms par requête sur CPU
- **Licence Apache 2.0** - utilisation commerciale complète

### Benchmarks (inférence CPU)

| Modèle | Params | MTEB | Latence CPU | Top-5 Acc | Multilingue |
|--------|--------|------|-------------|-----------|-------------|
| e5-small | 118M | - | 16ms | 100% | Limité |
| **BGE-M3** | 567M | 63.0 | <30ms | Compétitif | **100+ langues** |
| Qwen3-Embed-8B | 8B | 70.58 | ~200ms | Élevée | 100+ langues |
| all-MiniLM-L6-v2 | 22.7M | - | 12ms | 56% | Anglais seul |

Pour une page PDF (~500-1000 tokens), BGE-M3 chunke et embed en **<5 secondes sur CPU**, bien dans notre cible de 10 secondes.

### Optimisations possibles

- **ONNX Runtime** - réduction de 30-40% de la latence via optimisation O3/O4
- **OpenVINO INT8** - quantification statique pour CPU Intel
- **Batch processing** - traitement par lots de 8 chunks simultanés

### Références

- Chen et al. (2024). *"BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"*. arXiv:2402.03216
- Muennighoff et al. (2023). *"MTEB: Massive Text Embedding Benchmark"*. EACL 2023
- AIM Research (2025). *"Benchmark of 16 Best Open Source Embedding Models for RAG"*

---

## 4. Base Vectorielle : Qdrant

### Justification

- **Natif Rust** - binaire compilé, overhead mémoire minimal, pas de pauses GC
- **Indexation HNSW** - Hierarchical Navigable Small World graphs pour recherche ANN sub-10ms
- **Quantification scalaire INT8** - réduction 4× de la RAM avec <1% de perte de recall
- **Filtrage payload** - index keyword sur `source_type`, `source_name` pour pré-filtrage O(1)
- **Self-hosted** - souveraineté des données, critique pour documents sensibles et résidence des données au Japon
- **Communauté active** - 21k+ étoiles GitHub, releases hebdomadaires, documentation excellente
- **API gRPC + REST** - client Python avec support async, opérations batch

### Configuration HNSW

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `m` | 16 | Chaque nœud connecté à 16 voisins. Bon compromis recall/mémoire |
| `ef_construct` | 200 | Haute qualité de construction. Indexation plus lente mais meilleur recall |
| `ef_search` | 128 | Largeur de faisceau à la recherche. Garantit >95% recall en sub-10ms |
| Quantification | INT8 scalar | Réduit vecteurs float32 1024-dim de 4KB à 1KB chacun |
| `always_ram` | True | Avec 64 Go RAM, on garde tous les vecteurs quantifiés en mémoire |

### Alternatives étudiées

| Base | Pour | Contre |
|------|------|--------|
| **Qdrant** | Rust, rapide, HNSW, quantification, filtrage | Process séparé (Docker) |
| PGVector | Intégration PostgreSQL | ANN plus lent, pas de quantification native |
| ChromaDB | API Python simple | Scaling limité, pas de quantification, SQLite backend |
| Weaviate | GraphQL, modules | Empreinte RAM plus lourde, Go-based |
| FAISS | In-process, rapide | Pas de persistance, pas de filtrage, pas d'API |

### Références

- Malkov & Yashunin (2018). *"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"*. IEEE TPAMI
- Documentation Qdrant - qdrant.tech/documentation/guides/optimize/
- Baranchuk et al. (2022). *"Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors"*

---

## 5. Pipeline RAG & Reranking

### Architecture

Le pipeline suit le pattern RAG canonique de Lewis et al. (2020), adapté pour l'inférence CPU-only :

1. **Ingestion** - PDF/DOCX/TXT/MD → extraction PyMuPDF → chunking par limites de phrases (512 chars, 64 overlap) → embedding BGE-M3 → upsert Qdrant avec métadonnées
2. **Requête** - Question utilisateur → embedding BGE-M3 → recherche ANN Qdrant (top-8, seuil 0.45) → **reranking cross-encoder (top-3)** → assemblage contexte avec tracking sources
3. **Génération** - Contexte + question + contexte personnel → Phi-4 avec prompt système orienté citations → réponse avec citations [Source: nom, page]

### Stratégie de chunking

- **Chunks de 512 caractères** - s'intègre bien dans la longueur max de séquence BGE-M3 (512 tokens)
- **Overlap de 64 caractères** - empêche la perte d'information aux frontières de chunks
- **Respect des limites de phrases** - découpe aux fins de phrases pour préserver la cohérence sémantique

### Mécanisme de citations

Chaque chunk récupéré porte des métadonnées (`source_name`, `page`, `chunk_index`). Le prompt système instruit le LLM de citer les sources au format `[Source: nom, p.X]`. Les sources sont également retournées en JSON structuré pour l'affichage frontend avec toggle par source.

### Reranker : cross-encoder/ms-marco-MiniLM-L-6-v2

- **Seulement 22M paramètres** - extrêmement rapide sur CPU (~5-15ms par paire query-document)
- **Entraîné sur MS MARCO** - 500M+ paires query-passage, le standard pour le passage reranking
- **NDCG@10 de 39.01** sur TREC Deep Learning 2019 - meilleur ratio vitesse/qualité pour CPU
- **API CrossEncoder sentence-transformers** - intégration directe, pas de dépendances supplémentaires

### Pipeline complet avec latences

| Étape | Modèle | Latence (CPU) | Sortie |
|-------|--------|--------------|--------|
| 1. Embedding | BGE-M3 | <30ms | Vecteur 1024-dim |
| 2. Recherche ANN | Qdrant HNSW | <10ms | Top-8 candidats |
| 3. Reranking | MiniLM-L-6-v2 | ~40-120ms (8 paires) | Top-3 reranked |
| 4. Génération | Phi-4 14B | 5-50s | Réponse avec citations |

### Limites identifiées

- **Pas de recherche hybride** - BGE-M3 supporte les vecteurs sparse mais la recherche hybride Qdrant ajoute de la complexité
- **Fenêtre de contexte** - 4096 tokens limite à ~3 chunks reranked par requête

### Références

- Lewis et al. (2020). *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*. NeurIPS 2020
- Gao et al. (2024). *"Retrieval-Augmented Generation for Large Language Models: A Survey"*. arXiv:2312.10997
- Nogueira & Cho (2019). *"Passage Re-ranking with BERT"*. arXiv:1901.04085
- Wang et al. (2020). *"MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression"*. NeurIPS 2020

---

## 6. Modèle de Vision : MiniCPM-V 2.6 (INT4)

### Justification

- **8B paramètres** (SigLip-400M encodeur vision + Qwen2-7B modèle langage) - compact pour un VLM
- **Quantification INT4** - ~4 Go RAM, s'intègre dans notre budget de 64 Go
- **Précision niveau GPT-4V** sur les benchmarks OCR, compréhension de documents et charts
- **Chargement à la demande** - chargé uniquement pour le traitement de PDFs visuels, déchargé pour libérer la RAM
- **Licence Apache 2.0**

### Utilisation dans MIRROR

Le modèle de vision est **optionnel** (activé via `VISION_ENABLED=1`). Quand activé, les pages PDF sont rendues à 150 DPI et analysées par le VLM. La description visuelle est ajoutée à l'extraction texte, enrichissant le contexte RAG. Sur CPU, chaque analyse de page prend ~30-60 secondes - acceptable pour le traitement asynchrone de documents.

### Référence

- Yao et al. (2024). *"MiniCPM-V: A GPT-4V Level MLLM on Your Phone"*. arXiv:2408.01800

---

## 7. Docker : Encapsulation Complète

L'ensemble du stack tourne via `docker compose up` - zéro dépendance hôte au-delà de Docker.

### Décisions de conception

- **Pas de limites CPU/mémoire fixes** - le scheduler cgroup de Docker gère la contention naturellement. Le nombre de threads est auto-détecté via `multiprocessing.cpu_count()`
- **Healthcheck Qdrant** - le conteneur app attend que Qdrant soit sain avant de démarrer
- **Volumes montés** - `./models`, `./uploads`, `./articles` montés depuis l'hôte pour faciliter le swap de modèles et la persistance des données
- **Un seul worker + 4 threads** - Gunicorn tourne avec 1 worker (le LLM est single-threaded) et 4 threads pour le handling HTTP concurrent

---

## 8. Web Scraping

- **trafilatura** (extraction principale) - précision >90% sur les benchmarks d'extraction web (Barbaresi, ACL 2021)
- **BeautifulSoup** (fallback) - parsing HTML robuste pour les cas non gérés par trafilatura
- Le contenu scrapé peut être interrogé directement (en mémoire) ou indexé dans Qdrant

### Référence

- Barbaresi (2021). *"Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction"*. ACL 2021

---

## 9. Bases de Données

### Architecture de stockage

MIRROR utilise deux bases de données complémentaires :

| Base | Type | Rôle | Scaling |
|------|------|------|---------|
| **SQLite** | Relationnelle embarquée | Conversations, logs, métadonnées utilisateur | Vertical (fichier unique) |
| **Qdrant** | Vectorielle (Rust) | Embeddings, recherche sémantique RAG | Horizontal (sharding natif) |

### Vertical vs Horizontal

- **Scaling vertical** - augmenter les ressources d'un serveur unique (CPU, RAM, SSD). Simple, pas de code distribué. Limité par le hardware disponible. Adapté à SQLite, PostgreSQL, MySQL pour des charges modérées.
- **Scaling horizontal** - distribuer les données sur plusieurs nœuds. Scaling quasi-infini, tolérance aux pannes. Complexité accrue (sharding, réplication, consistance). Adapté à MongoDB, Cassandra, Qdrant, Kafka.
- **Théorème CAP** - un système distribué ne peut garantir que 2 des 3 : Consistance, Disponibilité, Tolérance au partitionnement. En pratique : choix entre CP (PostgreSQL, CockroachDB) et AP (Cassandra, DynamoDB).

### Pourquoi pas PostgreSQL ?

Pour un portfolio mono-utilisateur, SQLite offre zéro configuration, backup trivial (copie du fichier), et performances excellentes en lecture. PostgreSQL serait le choix pour une application multi-utilisateurs avec concurrence d'écriture.

---

## 10. CI/CD & Monitoring

### Pipeline de déploiement

| Outil | Rôle | Justification |
|-------|------|---------------|
| **GitHub Actions** | CI/CD | Lint, tests, build Docker, push registry. Gratuit pour open source. |
| **Docker Compose** | Déploiement | Stack complète en une commande. Adapté au mono-serveur. |
| **Caddy** | Reverse proxy | HTTPS automatique (Let's Encrypt), HTTP/2, config minimale. |

### Monitoring (cibles futures)

| Outil | Rôle |
|-------|------|
| **Prometheus** | Collecte de métriques (latence, tokens/s, erreurs, RAM/CPU) |
| **Grafana** | Dashboards et alertes visuelles |
| **Loki** | Agrégation de logs (intégré Grafana) |
| **Langfuse** | Tracing LLM spécifique (prompt, contexte, réponse, latence) |

### Orchestration de pipelines ML

- **Apache Airflow** - orchestrateur de workflows Python (DAGs). Standard pour ETL, pipelines d'entraînement, validation de données. Alternatives : Prefect (DX moderne), Dagster (asset-centric).
- **Snowflake** - data warehouse cloud avec séparation stockage/compute. Snowpark pour ML en Python directement dans le warehouse. Cortex AI pour LLM en SQL.

---

## 11. Stack Technique Complet

| Couche | Technologie | Rôle |
|--------|------------|------|
| Framework Web | Flask 3.1 | Templates Jinja2, architecture blueprints |
| LLM | Phi-4 14B Q4_K_M + llama-cpp-python | Inférence CPU, ~5-10 t/s, 9 Go RAM |
| Embedding | BGE-M3 + sentence-transformers | Multilingue, 1024-dim, <30ms/requête |
| Reranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder, 22M params, ~15ms/paire sur CPU |
| Vision (opt.) | MiniCPM-V 2.6 INT4 | Compréhension visuelle PDF, ~4 Go RAM |
| Base Vectorielle | Qdrant (Docker) | HNSW + quantification INT8, recherche sub-10ms |
| Base Relationnelle | SQLite (WAL) | Conversations, logs, métadonnées |
| Parsing PDF | PyMuPDF (fitz) | Rapide, gère les layouts complexes |
| Scraping | trafilatura + BeautifulSoup | Extraction haute précision |
| Reverse Proxy | Caddy 2.8 | HTTPS auto, HTTP/2, config minimale |
| Conteneurisation | Docker Compose | Encapsulation complète, portable |
| CI/CD | GitHub Actions | Lint, test, build, deploy |

---

## 12. Optimisations Futures

1. **ONNX Runtime** pour BGE-M3 - gagner 30-40% de vitesse d'embedding
2. **Streaming SSE** - afficher les tokens en temps réel pendant la génération
3. **Reranking conditionnel** - activer le cross-encoder uniquement quand le top-1 score < 0.7
4. **Cache de requêtes** - Redis/mémoire pour les questions fréquentes
5. **Recherche hybride Qdrant** - combiner dense + sparse quand le corpus dépasse 10k chunks
6. **GPU futur** - si GPU disponible, migrer vers vLLM pour 50-100× le throughput
7. **Vision enrichie** - intégrer la description visuelle automatiquement dans le pipeline d'ingestion PDF
