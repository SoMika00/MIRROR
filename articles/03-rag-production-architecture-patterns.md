---
title: "RAG en Production : Patterns d'Architecture et Retours d'Expérience"
date: 2025-01-10
tags: RAG, Architecture, Vector DB, Embeddings, Production
summary: "Retour d'expérience sur la mise en production d'un pipeline RAG complet : chunking, embeddings, reranking, orchestration, et les pièges à éviter pour passer du prototype au système fiable."
---

# RAG en Production : Patterns d'Architecture et Retours d'Expérience

## Introduction

Le RAG (Retrieval Augmented Generation) est devenu le pattern dominant pour connecter un LLM à des données métier. Mais le fossé entre un prototype Jupyter Notebook et un système de production fiable est immense. Cet article documente les patterns d'architecture éprouvés et les erreurs classiques rencontrées lors de la mise en production de pipelines RAG.

## 1. Anatomie d'un Pipeline RAG Robuste

Un pipeline RAG de production se décompose en deux phases distinctes : l'**ingestion** (offline) et le **serving** (online, temps réel).

### 1.1 Phase d'Ingestion (Offline)

L'ingestion transforme vos documents bruts en vecteurs indexés :

- **Extraction** : PDF, DOCX, HTML, Markdown → texte brut. Les outils varient : `unstructured.io` pour les cas complexes (tableaux, images), `pypdf2` ou `pdfplumber` pour les cas simples.
- **Chunking** : Découpage du texte en passages de 256-512 tokens avec un overlap de 10-20%. Le choix de la taille du chunk est critique : trop petit = perte de contexte, trop grand = bruit dans les résultats.
- **Embedding** : Chaque chunk est transformé en vecteur dense via un modèle d'embedding (BGE-M3, E5-Large, GTE). Le choix du modèle d'embedding impacte directement la qualité du retrieval.
- **Indexation** : Les vecteurs sont stockés dans une base vectorielle (Qdrant, Pinecone, Weaviate) avec leurs métadonnées (source, page, date).

### 1.2 Phase de Serving (Online)

Le serving traite les requêtes utilisateur en temps réel :

1. **Reformulation** : Optionnelle mais efficace. Un petit LLM reformule la question de l'utilisateur pour améliorer le recall.
2. **Retrieval** : Recherche des top-k chunks les plus similaires via ANN (Approximate Nearest Neighbor).
3. **Reranking** : Un cross-encoder (BGE-Reranker, Cohere Rerank) re-classe les résultats pour maximiser la précision.
4. **Génération** : Le LLM reçoit la question + les chunks pertinents et génère une réponse avec citations.

## 2. Stratégies de Chunking Avancées

Le chunking naïf (découpage tous les N tokens) est rarement optimal en production.

### Chunking Sémantique

Plutôt que de couper à une taille fixe, on détecte les frontières sémantiques naturelles : fins de paragraphes, changements de sujet, titres de sections. L'implémentation repose souvent sur un sliding window d'embeddings : quand la similarité cosine entre deux fenêtres adjacentes chute sous un seuil, on coupe.

### Chunking Hiérarchique

On maintient deux niveaux : des chunks fins (256 tokens) pour le retrieval précis, et des chunks larges (1024 tokens) pour le contexte. Le retrieval se fait sur les chunks fins, mais le contexte envoyé au LLM inclut le chunk large parent.

### Chunking par Document Structure

Pour les documents structurés (HTML, Markdown, PDF avec titres), on respecte la structure du document : un chunk = une section. Les métadonnées de hiérarchie (titre parent, numéro de section) sont conservées.

## 3. Le Piège de l'Évaluation

Évaluer un pipeline RAG est notoirement difficile car il faut évaluer **deux composants séparément** :

- **Retrieval Quality** : Les bons documents sont-ils récupérés ? Métriques : Recall@k, MRR, NDCG.
- **Generation Quality** : La réponse est-elle correcte et fidèle aux documents ? Métriques : faithfulness, answer relevancy, hallucination rate.

### Framework RAGAS

RAGAS propose des métriques automatisées : **Faithfulness** (la réponse est-elle supportée par le contexte ?), **Answer Relevancy** (la réponse répond-elle à la question ?), **Context Precision** (les chunks récupérés sont-ils pertinents ?).

## 4. Patterns de Production

### Hybrid Search

Combiner la recherche vectorielle (dense) avec la recherche lexicale (sparse, BM25). Les requêtes contenant des termes techniques précis (codes produit, noms propres) bénéficient énormément du BM25 en complément.

### Query Routing

Toutes les questions ne nécessitent pas le RAG. Un classifier léger peut router les questions générales vers le LLM seul, et les questions spécifiques vers le pipeline RAG. Cela réduit la latence et le coût.

### Metadata Filtering

Filtrer les résultats par métadonnées avant le ranking vectoriel : date, département, type de document. Cela réduit l'espace de recherche et améliore la précision.

## 5. Monitoring et Observabilité

En production, surveiller :

- **Retrieval latency** : p50, p95, p99
- **LLM generation latency** : time to first token, total generation time
- **Retrieval quality** : taux de questions sans résultats pertinents (empty retrieval)
- **User feedback** : thumbs up/down sur les réponses
- **Token usage** : coût par requête, context window utilization

## Conclusion

Un pipeline RAG de production est un système distribué complexe qui nécessite une attention particulière au chunking, à l'évaluation, et au monitoring. Les patterns décrits ici sont le fruit d'itérations successives sur des déploiements réels. Le plus important : commencer simple, mesurer, itérer.
