---
title: "RAG en Production : Construire un Systeme Complet sur un Seul Serveur"
date: 2025-01-31
tags: RAG, Architecture, Vector DB, Embeddings, Production, Qdrant, BGE-M3, Reranking, Infrastructure
summary: "Retour d'experience sur la construction d'un pipeline RAG de bout en bout pour MIRROR - architecture, choix techniques, infrastructure bare-metal, TLS, reverse proxy, et les patterns qui ont survecu a la production sur un seul serveur CPU."
---

# RAG en Production : Construire un Systeme Complet sur un Seul Serveur

## Le contexte

MIRROR est un systeme RAG complet qui tourne sur un serveur dedie Hetzner (12 vCPU, 64 Go RAM, zero GPU). L'utilisateur uploade des PDFs, des URLs, du Markdown, et le systeme repond en citant ses sources en francais, anglais ou japonais.

Cet article couvre deux choses : le pipeline RAG lui-meme (ce qui a marche, ce qui n'a pas marche) et l'infrastructure pour le faire tourner en production (serveur, TLS, reverse proxy, monitoring). C'est le guide que j'aurais voulu avoir avant de commencer.

## 1. L'architecture qui a survecu

Apres plusieurs iterations, voici le pipeline final :

```
Question utilisateur
    |
    v
[Query Router]  <-- classification legere (regex + heuristiques, <1ms)
    |
    |-- SIMPLE --> LLM direct (pas de RAG, ~100ms)
    |
    +-- MEDIUM/COMPLEX --> Pipeline RAG complet :
            |
            v
        [BGE-M3 Embedding]    -->  1024 dimensions, ~30ms
            |
            v
        [Qdrant Search]       -->  top-12, seuil 0.35, HNSW, ~15ms
            |
            v
        [Cross-Encoder Rerank] --> ms-marco-MiniLM-L-6-v2, seuil -3.0, ~80ms
            |
            v
        [Context Assembly]     --> top-5, scores de pertinence, budget tokens
            |
            v
        [Llama 3.1 8B Q4_K_M] --> 8192 ctx, mlock, ~10-18 t/s
            |
            v
        Reponse avec sources separees
```

Chaque composant a ete choisi apres avoir teste des alternatives.

## 2. L'embedding : pourquoi BGE-M3

J'ai teste `E5-large-v2`, `GTE-large`, `all-MiniLM-L6-v2`, et `BGE-M3`. Critere principal : la qualite multilingue (francais + anglais + japonais dans le meme index).

`all-MiniLM-L6-v2` est rapide mais ses embeddings en francais sont mediocres : recall@10 tombait a 0.6 sur mes tests, contre 0.85 en anglais. `E5-large-v2` est correct en multi-langue mais ses vecteurs de 1024 dims prenaient trop de RAM pour la meme qualite que BGE-M3.

BGE-M3 a gagne pour trois raisons :
1. Entraine nativement en 100+ langues, meme performance en francais, anglais, japonais
2. Supporte les passages longs (jusqu'a 8192 tokens), crucial pour des chunks de 768 tokens
3. Dense + sparse dans le meme modele (je n'utilise que le dense, mais l'option est la)

Le piege : BGE-M3 prend ~2.5 Go de RAM. Sur un serveur a 64 Go ou le LLM prend 3-9 Go et Qdrant 2-4 Go, chaque Go compte. Il faut mesurer precisement la consommation de chaque composant.

## 3. Chunking : les 768 tokens qui ont tout change

Mon premier chunking etait a 512 tokens avec 64 d'overlap. C'est ce que recommandent les tutos. C'etait mauvais.

Le probleme : mes documents techniques ont des paragraphes denses. A 512 tokens, on coupe au milieu d'une explication. Le LLM recoit un bout de phrase, ne comprend pas le contexte, et hallucine pour combler.

J'ai teste systematiquement 256, 512, 768, et 1024 tokens sur un jeu de 50 questions-reponses manuelles :

| Chunk size | Recall@5 | Faithfulness | Hallucination rate |
|-----------|----------|-------------|-------------------|
| 256       | 0.72     | 0.68        | 23%               |
| 512       | 0.78     | 0.74        | 18%               |
| **768**   | **0.83** | **0.81**    | **11%**           |
| 1024      | 0.80     | 0.79        | 13%               |

768 tokens etait le sweet spot : assez long pour conserver le contexte, assez court pour que le retrieval reste precis. L'overlap de 128 tokens (17%) garantit qu'aucune phrase importante ne tombe entre deux chunks.

Lecon : ne faites pas confiance aux valeurs par defaut. Testez sur vos donnees avec vos questions.

## 4. Le reranker : l'arme secrete sous-estimee

Sans reranker, le pipeline retournait des resultats "proches mais pas pertinents". La similarite cosine est bonne pour le recall, mauvaise pour la precision.

Exemple concret : pour la question "Quelle est l'architecture de MIRROR ?", le retrieval dense retournait :
1. Un chunk sur l'architecture de MIRROR (pertinent)
2. Un chunk sur l'architecture des Transformers (non pertinent : cosine sim haute car "architecture" est un terme dominant)
3. Un chunk sur les choix d'infrastructure de MIRROR (pertinent)

Le cross-encoder `ms-marco-MiniLM-L-6-v2` resout ca en ~80ms. Il prend la question + chaque chunk et produit un score de pertinence contextuel, pas juste une distance vectorielle.

Mon setup : retrieve top-12 (large net), puis le reranker garde les top-5 avec un seuil a -3.0 (tout ce qui est en dessous est du bruit). Le passage de top-3 a top-5 a ete rendu possible par l'augmentation du context window a 8192 tokens.

Le piege du reranker : il charge un modele BERT de ~80 Mo en RAM et ajoute ~80ms de latence. Sur des requetes simples (salutations, questions personnelles), c'est du gaspillage. D'ou le query router.

## 5. Le Query Router : ne pas tout traiter pareil

Le query router classe les requetes en trois tiers sans appel LLM, juste des regex et des heuristiques :

- **SIMPLE** (salutations, questions personnelles) : LLM direct, pas de RAG, ~100ms
- **MEDIUM** (questions factuelles) : RAG standard, ~2-5s
- **COMPLEX** (comparaisons, multi-parties, technique profond) : RAG etendu avec plus de chunks, ~5-15s

Le routage prend <1ms. Sur un mix de requetes realiste, ca reduit la latence moyenne de ~40% parce que la majorite des interactions sont des salutations ou des questions simples.

Le code est ~150 lignes de Python. Pas de ML, pas de classifier entraine. Des listes de mots-cles, des regex, et de la logique conditionnelle. Parfois la solution la plus simple est la meilleure.

## 6. Le prompt : l'iteration invisible

Le prompt RAG a traverse une dizaine de versions. Chaque instruction corrige un bug observe :

- Sans "Read ALL documents", le modele lisait le premier chunk et ignorait les autres
- Sans "Answer in the SAME language", il repondait en anglais a une question en francais
- Sans "do NOT hallucinate", il inventait des reponses plausibles mais fausses
- Sans instruction sur les sources, il paraphrasait sans citer, rendant la verification impossible

Un bon prompt RAG est un document de requirements. Chaque ligne corrige un bug observe en production.

## 7. Le context assembly

Assembler le contexte pour le LLM n'est pas juste "concatener les chunks" :

- **Tags de pertinence** : chaque chunk est annote `[Relevance: HIGH/MEDIUM/LOW]` avec le score numerique. Le LLM peut ponderer.
- **Budget tokens dynamique** : le context est tronque pour respecter `n_ctx - max_tokens - prompt_overhead`. Pas de troncation aveugle, on coupe au dernier chunk complet.
- **Ordre par pertinence** : les chunks les plus pertinents en premier. Les LLMs ont un biais de position (primacy effect), mettez l'important au debut.

## 8. L'infrastructure : de zero a la production

C'est la partie que personne ne documente. Voici comment j'ai monte MIRROR de A a Z sur un seul serveur.

### 8.1 Le serveur

Hetzner AX102 dedie : 12 cores, 64 Go RAM DDR4, 2x 512 Go NVMe. Cout : ~65 euros/mois. Pas de GPU. Le choix est delibere : prouver qu'un RAG complet peut tourner sur du hardware modeste.

Pourquoi un serveur dedie et pas du cloud (AWS/GCP) ? Parce que pour un projet personnel avec du LLM inference qui tourne en continu, le dedie est 3-5x moins cher que des instances cloud equivalentes. Pas d'auto-scaling necessaire, pas de latence reseau entre services.

### 8.2 Le nom de domaine et le DNS

Configuration DNS :
- Record A pointant vers l'IP du serveur
- Record AAAA si IPv6 disponible
- TTL de 300s pour pouvoir migrer rapidement

### 8.3 TLS et reverse proxy avec Caddy

Caddy est le choix ideal pour un serveur solo :

```
mondomaine.com {
    reverse_proxy /api/* localhost:5000 {
        flush_interval -1
    }
    reverse_proxy localhost:5000 {
        flush_interval -1
    }
}
```

Caddy gere automatiquement :
- Certificat TLS via Let's Encrypt (zero configuration)
- Renouvellement automatique tous les 60 jours
- HTTP/2 et HTTP/3 par defaut
- Redirection HTTP vers HTTPS

Le `flush_interval -1` est critique pour le streaming SSE (Server-Sent Events). Sans ca, Caddy bufferise les reponses et le streaming ne fonctionne pas.

### 8.4 Docker Compose : tout orchestrer

Trois services dans un seul `docker-compose.yml` :

```yaml
services:
  caddy:        # Reverse proxy + TLS
  qdrant:       # Vector DB
  flask-app:    # Application (LLM + embedding + reranker)
```

Points critiques :
- `mlock: true` dans les options llama.cpp pour empecher le swap du modele
- Volume persistant pour Qdrant (les embeddings survivent aux redemarrages)
- Volume pour les certificats Caddy
- Health checks sur chaque service
- Restart policy `unless-stopped`

### 8.5 Monitoring en production

Metriques exposees via `/api/models/metrics` :
- **embed_time** : temps d'embedding de la question (~30ms)
- **search_time** : temps de recherche Qdrant (~15ms)
- **rerank_time** : temps du cross-encoder (~80ms)
- **generation_time** : temps de generation LLM (2-15s selon complexite)
- **tokens/sec** : vitesse de generation en temps reel
- CPU par coeur, RAM, RSS du process, affichage live dans l'interface

## 9. Ce qui n'a PAS marche

### Semantic Query Cache
Cacher les reponses par similarite d'embedding. En theorie, gain de latence massive. En pratique, avec l'historique qui change le contexte, les hits etaient rares et les faux positifs dangereux.

### Chunking semantique par embeddings
Couper quand la similarite cosine entre fenetres adjacentes chute. Trop lent a l'ingestion, resultats pas significativement meilleurs que le chunking fixe a 768 tokens.

### Query expansion avec LLM
Reformuler la question avec un appel LLM avant le retrieval. +5% recall, mais +2-3s de latence. Sur CPU, chaque seconde compte. Le reranker apporte un meilleur gain pour moins de cout.

## Conclusion

Un RAG de production, ce n'est pas un pipeline lineaire qu'on branche et qui marche. C'est un systeme avec des dizaines de parametres interdependants qu'il faut calibrer sur ses propres donnees. Et c'est aussi une stack infra complete : serveur, DNS, TLS, reverse proxy, containers, monitoring.

Trois choses que j'aurais aime savoir avant de commencer :
1. **Le chunking est le parametre le plus impactant** : plus que le modele d'embedding, plus que le LLM
2. **Le reranker vaut chaque milliseconde** : c'est le meilleur rapport qualite/cout du pipeline
3. **L'infrastructure est 50% du travail** : un bon RAG sur une mauvaise infra ne sert a rien

*Michail Berjaoui - Janvier 2025*
