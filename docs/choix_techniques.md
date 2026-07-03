# MIRROR - Choix Techniques (V2, API-first)

## Document de référence pour l'architecture du portfolio IA de Michail Berjaoui

---

## 0. Résumé

MIRROR est une application RAG de production, pas un portfolio statique. La
génération passe par l'**API xAI Grok**, la retrieval est **hybride BM25
(SQLite FTS5)** avec embeddings par API en option, l'état vit dans **SQLite
(WAL)**, et le tout est déployé avec **Docker Compose + Caddy** (TLS
automatique) sur un petit VPS. Aucun poids de modèle ne tourne sur le serveur,
et un budget journalier dur plafonne la dépense LLM sous **0,50 $/jour**.

| Couche | Choix | Pourquoi |
|--------|-------|----------|
| LLM | API xAI Grok (grok-4.20 non-reasoning) | Qualité frontier, ~1 s au premier token, zéro RAM, paiement à l'usage |
| Retrieval | SQLite FTS5 · BM25, prêt pour l'hybride | In-process, zéro service à opérer, dimensionné pour le corpus |
| Embeddings | Optionnels, via API (compatible OpenAI) | Aucun modèle local ; l'hybride dense+lexical s'active avec une variable d'env |
| État applicatif | SQLite (WAL) | Un fichier, atomique, zéro ops |
| Serving | Flask + gunicorn (2 workers × 8 threads) | Charge I/O-bound : les threads attendent l'API, pas le CPU |
| Edge | Caddy 2 | HTTPS automatique Let's Encrypt, config de 10 lignes |
| Hébergement | Petit VPS, Docker Compose | Coût total infra + inférence : moins de ~1 $/jour |

---

## 1. De la stack souveraine à l'API-first : une décision d'architecture

La décision d'ingénierie la plus importante du projet a été de **tuer ma
propre première architecture**. La V1 tournait en stack souveraine complète
(Phi-4 14B via llama.cpp, BGE-M3, CrossEncoder, Qdrant - ~15 Go de RAM sur un
serveur 64 Go). La V2 tourne entièrement sur API.

Les deux patterns sont légitimes - j'ai livré les deux professionnellement :
LLM souverain sur Scaleway pour un client entreprise avec contraintes de
résidence des données ; API-first quand la vitesse d'itération prime.

Pourquoi la bascule était juste **pour ce produit** :

- **L'expérience utilisateur est le produit.** Un recruteur attend 2 secondes,
  pas 20. L'inférence CPU d'un 14B ne tenait pas cette barre.
- **La qualité signale la compétence.** Un 14B quantifié imprécis sur mon
  propre CV était une pire vitrine qu'un modèle frontier qui répond bien.
- **Le coût s'adapte au trafic.** Un portfolio a un trafic faible et
  sporadique. Payer 64 Go de RAM 24/7 pour quelques conversations par jour est
  la mauvaise forme de coût ; le paiement au token épouse exactement le profil.
- **Aucune contrainte de souveraineté ici.** Le corpus est mon CV public et
  mes articles publiés. Quand la donnée est confidentielle, l'arbitrage
  s'inverse - et j'ai déployé la version souveraine de cette même stack pour
  ce cas.

---

## 2. LLM : API Grok avec garde-fou de coût

- Endpoint compatible OpenAI de xAI, modèle `grok-4.20-non-reasoning` : pour
  des réponses conversationnelles courtes, un modèle "reasoning" brûle des
  tokens de sortie (et des secondes) sans améliorer une réponse de CV.
- **Budget journalier dur** (0,50 $/jour par défaut), persisté sur disque,
  vérifié **avant** chaque appel - un chatbot public sans plafond de dépense
  est un portefeuille ouvert.
- **Comptabilité réelle** : les comptes de tokens viennent du champ `usage` de
  l'API (y compris en streaming via `stream_options.include_usage`), pas
  d'estimations ; les prix sont de la configuration, pas des constantes.
- **Dégradation gracieuse** : plafond atteint → l'assistant l'annonce et
  revient demain. Le reste du site n'est pas affecté.
- **Discipline de contexte** : contexte de retrieval plafonné (~6K tokens),
  historique tronqué - le premier facteur de coût d'un RAG est l'assemblage
  de contexte non borné.

Ordre de grandeur : un tour RAG typique ≈ 3K tokens in + 400 out ≈ 0,005 $ ;
le plafond de 0,50 $ couvre ≈ 100 conversations RAG/jour.

---

## 3. Retrieval : BM25 hybride, dimensionné juste

Le corpus (CV, 7 articles techniques en 2 langues, docs d'architecture,
uploads visiteurs) fait quelques centaines de chunks. À cette échelle, une
base vectorielle dédiée est de la sur-ingénierie.

- **Indexation** : chunks de 768 caractères (overlap 128, découpe aux limites
  de phrases) dans une table `chunks` ; table virtuelle FTS5 (tokenizer
  unicode61, insensible aux diacritiques - important pour le français)
  synchronisée par triggers.
- **Requête** : texte libre assaini en match FTS5 OR ; scores `bm25()`
  ramenés dans (0,1].
- **Chemin d'upgrade hybride** : chaque chunk a un BLOB d'embedding optionnel.
  Si `EMBEDDINGS_API_KEY` est définie (tout fournisseur compatible OpenAI),
  les vecteurs sont récupérés à l'indexation et la recherche fusionne
  `0,6 × cosinus + 0,4 × BM25` sur un pool de candidats 4×. Une variable
  d'env, zéro migration. xAI n'expose pas encore de modèle d'embeddings ; le
  client est précâblé pour le jour où.
- **Scoping par utilisateur** : la connaissance portfolio est globale ; les
  uploads et pages scrapées sont scellés à une session cookie anonyme.
- **Auto-indexation** : au démarrage, le site indexe son propre contenu
  (profil, articles, ces docs) - l'assistant répond sur moi avec citations
  sans aucune action visiteur.

**Pourquoi ni reranker ni base vectorielle ici** : le reranking cross-encoder
paie sa latence quand le recall de premier étage est bruité sur un grand
corpus. Sur un corpus petit et maîtrisé, BM25 + un LLM frontier qui lit 8
chunks suffit - le LLM est le reranker. Qdrant (que j'opère en production
ailleurs : tuning HNSW, quantization INT8, index de payload) devient rentable
vers 10⁵-10⁶ vecteurs avec un vrai QPS. L'utiliser ici serait de
l'infrastructure pour le CV.

---

## 4. Pipeline RAG et contrôle du comportement

- **Routage adaptatif** : classifieur heuristique à latence nulle (longueur,
  type de question, densité de mots-clés) - salutations en chat direct,
  questions de fond en RAG. Aucun appel LLM dépensé pour router.
- **Prompting orienté citations** : chunks étiquetés par source ; le prompt
  système exige `[Source : ...]` et autorise explicitement « les documents ne
  couvrent pas ce point » - contrôle d'hallucination by design, vérifiable
  par l'utilisateur dans l'UI.
- **Streaming de bout en bout** : SSE de Flask à travers Caddy jusqu'au
  navigateur ; les sources sont émises en premier événement.
- **Chaîne de fallback** : échec de retrieval → chat direct avec le contexte
  personnel ; budget épuisé → message explicite ; erreur API → remontée dans
  le stream, jamais d'échec silencieux.
- **Modes supplémentaires** : upload PDF/DOCX/TXT/MD (PyMuPDF) ou scraping
  d'URL (trafilatura), interrogeables avec le même pipeline - pour que les
  visiteurs techniques testent un vrai RAG multi-tenant.

---

## 5. Données et sessions

| Table | Rôle |
|-------|------|
| `users` | Sessions anonymes (cookie UUID, TTL 1 an, HttpOnly, SameSite=Lax) |
| `conversations` / `messages` | Fils de discussion par utilisateur, sources et timings en JSON |
| `chunks` + `chunks_fts` | Store de retrieval : texte, métadonnées, embedding optionnel, index FTS5 |
| `user_sources` | Suivi des sources documents/web par utilisateur |
| `logs` | Logs applicatifs structurés, requêtables par API |

SQLite en WAL : lecteurs concurrents + un écrivain, exactement le profil de
cette charge ; connexions thread-local pour éviter la contention entre
threads gunicorn.

---

## 6. Déploiement

- **Docker Compose, 2 services** : `mirror` (Flask/gunicorn) + `caddy`. La V1
  avait 3 services et une image de ~8 Go (torch + poids) ; l'image V2 fait
  ~450 Mo.
- **Caddy 2** : HTTPS automatique (Let's Encrypt), headers de sécurité
  (nosniff, frame-deny, referrer-policy) à l'edge.
- **12-factor** : tout passe par `.env` - modèle, prix, budget, clés.
- **Publication d'article** = déposer un fichier markdown dans `articles/`,
  indexé au prochain démarrage.

---

## 7. Ce qui change à l'échelle (et ce que je garderais)

| Dimension | Ici (portfolio) | À l'échelle (ce que je livre aux clients) |
|-----------|-----------------|-------------------------------------------|
| Retrieval | FTS5 BM25, prêt hybride | Qdrant/PGVector, hybride dense+sparse, reranking cross-encoder |
| Inférence | API Grok, budget plafonné | vLLM sur GPU (continuous batching, paged attention), autoscaling SLO |
| Évaluation | Vérifications manuelles | Jeux d'éval offline (recall, faithfulness), LLM-as-judge en CI |
| Observabilité | Logs structurés + table SQLite | Prometheus/Grafana, traces par étage, coût par tenant |
| État | SQLite WAL | PostgreSQL, isolation par tenant, migrations |
| Delivery | Compose sur un VPS | Kubernetes + Helm + Terraform, blue-green |

Le dimensionnement juste coupe dans les deux sens : cette architecture est
volontairement minimale, et je sais exactement où elle cesse de suffire.
