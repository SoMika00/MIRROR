---
title: "RAG en Production : Ce que j'ai appris en construisant MIRROR"
date: 2025-01-10
tags: RAG, Architecture, Vector DB, Embeddings, Production, Qdrant, BGE-M3, Reranking
summary: "Retour d'expérience concret sur la construction d'un pipeline RAG complet pour MIRROR — mon portfolio IA qui tourne sur un seul serveur CPU. Chunking, embeddings, reranking, les erreurs qui m'ont coûté des jours, et les patterns qui ont survécu à la production."
---

# RAG en Production : Ce que j'ai appris en construisant MIRROR

## Le contexte

MIRROR est mon portfolio IA. C'est un système RAG complet qui tourne sur un serveur dédié Hetzner (12 vCPU, 64 Go RAM, **zéro GPU**). L'utilisateur uploade des PDFs, des URLs, du Markdown, et le système répond en citant ses sources — en français, anglais ou japonais.

Ça semble simple dit comme ça. Ça ne l'est pas.

Cet article n'est pas un tutoriel. C'est un retour d'expérience : les choix que j'ai faits, pourquoi, ce qui a marché, et surtout ce qui n'a pas marché au début.

## 1. L'architecture qui a survécu

Après plusieurs itérations, voici le pipeline final :

```
Question utilisateur
    │
    ▼
[Query Router]  ←── classification légère (regex + heuristiques, <1ms)
    │
    ├── SIMPLE → LLM direct (pas de RAG, ~100ms)
    │
    └── MEDIUM/COMPLEX → Pipeline RAG complet :
            │
            ▼
        [BGE-M3 Embedding]  →  1024 dimensions, ~30ms
            │
            ▼
        [Qdrant Search]     →  top-12, seuil 0.35, HNSW, ~15ms
            │
            ▼
        [Cross-Encoder Rerank]  →  ms-marco-MiniLM-L-6-v2, filtre <-3.0, ~80ms
            │
            ▼
        [Context Assembly]  →  top-5 résultats, scores de pertinence, budget tokens
            │
            ▼
        [Phi-4 Mini 3.8B]  →  Q4_K_M, 8192 ctx, mlock, ~15-25 t/s
            │
            ▼
        Réponse avec [Source: nom, p.X]
```

Chaque composant a été choisi après avoir testé des alternatives. Laissez-moi vous expliquer pourquoi.

## 2. L'embedding : pourquoi BGE-M3 et pas les autres

J'ai testé `E5-large-v2`, `GTE-large`, `all-MiniLM-L6-v2`, et `BGE-M3`. Mon critère principal : **la qualité multilingue** (français + anglais + japonais dans le même index).

`all-MiniLM-L6-v2` est rapide mais ses embeddings en français sont médiocres — recall@10 tombait à 0.6 sur mes tests, contre 0.85 en anglais. `E5-large-v2` est correct en multi-langue mais ses vecteurs de 1024 dims prenaient trop de RAM pour la même qualité que BGE-M3.

**BGE-M3** a gagné pour trois raisons :
1. Entraîné nativement en 100+ langues — même performance en français, anglais, japonais
2. Supporte les passages longs (jusqu'à 8192 tokens) — crucial pour des chunks de 768 tokens
3. Dense + sparse dans le même modèle (je n'utilise que le dense pour l'instant, mais l'option est là)

**Le piège** : BGE-M3 prend ~2.5 Go de RAM. Sur un serveur à 64 Go où le LLM prend 3-9 Go et Qdrant 2-4 Go, chaque Go compte. J'ai dû mesurer précisément la consommation de chaque composant avant de valider.

## 3. Chunking : les 768 tokens qui ont tout changé

Mon premier chunking était à 512 tokens avec 64 d'overlap. C'est ce que recommandent tous les tutos. **C'était mauvais.**

Le problème : mes documents techniques (articles, rapports) ont des paragraphes denses. À 512 tokens, on coupe souvent au milieu d'une explication. Le LLM reçoit un bout de phrase, ne comprend pas le contexte, et hallucine pour combler.

J'ai testé systématiquement 256, 512, 768, et 1024 tokens sur un jeu de 50 questions-réponses manuelles :

| Chunk size | Recall@5 | Faithfulness | Hallucination rate |
|-----------|----------|-------------|-------------------|
| 256       | 0.72     | 0.68        | 23%               |
| 512       | 0.78     | 0.74        | 18%               |
| **768**   | **0.83** | **0.81**    | **11%**           |
| 1024      | 0.80     | 0.79        | 13%               |

768 tokens était le sweet spot : assez long pour conserver le contexte, assez court pour que le retrieval reste précis. L'overlap de 128 tokens (17%) garantit qu'aucune phrase importante ne tombe entre deux chunks.

**Leçon** : ne faites pas confiance aux valeurs par défaut. Testez sur VOS données avec VOS questions.

## 4. Le reranker : l'arme secrète sous-estimée

Sans reranker, mon pipeline retournait des résultats "proches mais pas pertinents". La similarité cosine est bonne pour le recall, mauvaise pour la précision.

Exemple concret : pour la question "Quelle est l'architecture de MIRROR ?", le retrieval dense retournait :
1. Un chunk sur l'architecture de MIRROR (✅)
2. Un chunk sur l'architecture des Transformers (❌ — cosine sim haute car "architecture" est un terme dominant)
3. Un chunk sur les choix d'infrastructure de MIRROR (✅)

Le cross-encoder `ms-marco-MiniLM-L-6-v2` résout ça en ~80ms. Il prend la question + chaque chunk et produit un score de pertinence contextuel — pas juste une distance vectorielle.

**Mon setup** : je retrieve top-12 (large net), puis le reranker garde les top-5 avec un seuil à -3.0 (tout ce qui est en dessous est du bruit). Le passage de top-3 à top-5 a été rendu possible par l'augmentation du context window à 8192 tokens.

**Le piège du reranker** : il charge un modèle BERT de ~80 Mo en RAM et ajoute ~80ms de latence. Sur des requêtes simples (salutations, questions personnelles), c'est du gaspillage. D'où le query router.

## 5. Le Query Router : ne pas tout traiter pareil

C'est un ajout récent et c'est un game-changer. Mon query router classe les requêtes en trois tiers sans aucun appel LLM — juste des regex et des heuristiques sur les mots-clés :

- **SIMPLE** (salutations, questions personnelles) → LLM direct, pas de RAG → ~100ms
- **MEDIUM** (questions factuelles) → RAG standard → ~2-5s
- **COMPLEX** (comparaisons, multi-parties, technique profond) → RAG étendu avec plus de chunks → ~5-15s

Le routage prend <1ms. Sur un mix de requêtes réaliste, ça réduit la latence moyenne de ~40% parce que la majorité des interactions sont des salutations ou des questions simples sur moi.

Le code est ~150 lignes de Python. Pas de ML, pas de classifier entraîné. Des listes de mots-clés, des regex, et de la logique conditionnelle. Parfois la solution la plus simple est la meilleure.

## 6. Le prompt : l'itération invisible

Le prompt RAG a traversé une dizaine de versions. La version actuelle :

```
You are MIRROR, an AI assistant powering Michail Berjaoui's portfolio website.

INSTRUCTIONS:
1. Read ALL the document excerpts carefully before answering.
2. Synthesize information across multiple sources when relevant.
3. Cite every claim from documents using [Source: name, p.X] format.
4. Answer in the SAME language as the question.
5. If the documents don't answer the question, say so explicitly — do NOT hallucinate.
```

Chaque instruction est là parce que sans elle, le modèle faisait n'importe quoi :
- Sans (1), il lisait le premier chunk et ignorait les autres
- Sans (3), il paraphrasait sans citer, rendant la vérification impossible
- Sans (5), il inventait des réponses plausibles mais fausses
- Sans (4), il répondait en anglais à une question en français

**La leçon** : un bon prompt RAG est un document de requirements. Chaque ligne corrige un bug observé.

## 7. Le context assembly : le détail qui fait la différence

Assembler le contexte pour le LLM n'est pas juste "concaténer les chunks". J'ai ajouté :

- **Tags de pertinence** : chaque chunk est annoté `[Relevance: HIGH/MEDIUM/LOW]` avec le score numérique. Le LLM peut pondérer.
- **Budget tokens dynamique** : le context est tronqué pour respecter `n_ctx - max_tokens - prompt_overhead`. Pas de troncation aveugle — on coupe au dernier chunk complet, avec "..." si nécessaire.
- **Ordre par pertinence** : les chunks les plus pertinents en premier. Les LLMs ont un biais de position (primacy effect) — mettez l'important au début.

## 8. Ce qui n'a PAS marché

Quelques idées que j'ai essayées et abandonnées :

### Semantic Query Cache
L'idée : cacher les réponses par similarité d'embedding pour les requêtes similaires. En théorie, gain de latence massive. En pratique, avec l'historique de conversation qui change le contexte, les hits étaient rares et les faux positifs dangereux (réponse cached pour un contexte différent).

### Chunking sémantique par embeddings
Couper quand la similarité cosine entre fenêtres adjacentes chute. Trop lent à l'ingestion (embedding de chaque fenêtre), et les résultats n'étaient pas significativement meilleurs que le chunking fixe à 768 tokens. Le rapport effort/gain n'y était pas.

### Query expansion avec LLM
Reformuler la question avec un appel LLM avant le retrieval. Ça améliorait le recall de ~5% mais ajoutait 2-3s de latence. Sur CPU, chaque seconde compte. Le reranker apporte un meilleur gain pour moins de coût.

## 9. Monitoring : ce que je mesure

En prod, je log systématiquement :
- **embed_time** : temps d'embedding de la question (~30ms)
- **search_time** : temps de recherche Qdrant (~15ms)
- **rerank_time** : temps du cross-encoder (~80ms)
- **generation_time** : temps de génération LLM (variable, ~2-15s selon la complexité)
- **total_time** : bout en bout
- **tokens/sec** : vitesse de génération en temps réel (affiché live dans le dashboard)

Ces métriques sont exposées via un endpoint `/api/models/metrics` et affichées en temps réel dans l'interface — CPU par cœur, RAM, RSS du process, et tokens/sec pendant l'inférence.

## Conclusion

Un RAG de production, ce n'est pas un pipeline linéaire qu'on branche et qui marche. C'est un système avec des dizaines de paramètres interdépendants (chunk size, overlap, top-k, seuil de score, reranker threshold, prompt, context budget) qu'il faut calibrer sur ses propres données.

Les trois choses que j'aurais aimé savoir avant de commencer :
1. **Le chunking est le paramètre le plus impactant** — plus que le modèle d'embedding, plus que le LLM
2. **Le reranker vaut chaque milliseconde** — c'est le meilleur rapport qualité/coût du pipeline
3. **Mesurez tout, tout le temps** — sans métriques, vous optimisez à l'aveugle
