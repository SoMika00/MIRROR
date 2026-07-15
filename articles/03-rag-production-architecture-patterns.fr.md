---
title: "RAG en Production : Construire un Système Complet sur un Seul Serveur"
date: 2025-01-31
tags: RAG, Architecture, Vector DB, Embeddings, Production, Qdrant, BGE-M3, Reranking, Infrastructure
summary: "Retour d'expérience sur la construction d'un pipeline RAG de bout en bout pour MIRROR - architecture, choix techniques, infrastructure bare-metal, TLS, reverse proxy, et les patterns qui ont survécu à la production sur un seul serveur CPU."
---

# RAG en Production : Construire un Système Complet sur un Seul Serveur

> **Note (2026)** : cet article décrit la V1 de MIRROR, entièrement auto-hébergée. L'architecture a depuis évolué vers une stack API-first (API Grok + retrieval hybride BM25) - les raisons de cette migration sont documentées sur la page [Architecture](/tech). Les leçons ci-dessous (chunking, reranking, prompt, infra) restent valables pour tout RAG souverain.

## Le contexte

MIRROR est un système RAG complet qui tournait sur un serveur dédié Hetzner (12 vCPU, 64 Go RAM, zéro GPU). L'utilisateur uploade des PDF, des URL, du Markdown, et le système répond en citant ses sources en français, anglais ou japonais.

Cet article couvre deux choses : le pipeline RAG lui-même (ce qui a marché, ce qui n'a pas marché) et l'infrastructure pour le faire tourner en production (serveur, TLS, reverse proxy, monitoring). C'est le guide que j'aurais voulu avoir avant de commencer.

## 1. L'architecture qui a survécu

Après plusieurs itérations, voici le pipeline final :

```
Question utilisateur
    |
    v
[Query Router]  <-- classification légère (regex + heuristiques, <1ms)
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
        Réponse avec sources séparées
```

Chaque composant a été choisi après avoir testé des alternatives.

## 2. L'embedding : pourquoi BGE-M3

J'ai testé `E5-large-v2`, `GTE-large`, `all-MiniLM-L6-v2` et `BGE-M3`. Critère principal : la qualité multilingue (français + anglais + japonais dans le même index).

`all-MiniLM-L6-v2` est rapide mais ses embeddings en français sont médiocres : le recall@10 tombait à 0,6 sur mes tests, contre 0,85 en anglais. `E5-large-v2` est correct en multilingue mais ses vecteurs de 1024 dimensions prenaient trop de RAM pour la même qualité que BGE-M3.

BGE-M3 a gagné pour trois raisons :
1. Entraîné nativement sur 100+ langues, même performance en français, anglais, japonais
2. Supporte les passages longs (jusqu'à 8192 tokens), crucial pour des chunks de 768 tokens
3. Dense + sparse dans le même modèle (je n'utilise que le dense, mais l'option est là)

Le piège : BGE-M3 prend ~2,5 Go de RAM. Sur un serveur à 64 Go où le LLM prend 3-9 Go et Qdrant 2-4 Go, chaque Go compte. Il faut mesurer précisément la consommation de chaque composant.

## 3. Chunking : les 768 tokens qui ont tout changé

Mon premier chunking était à 512 tokens avec 64 d'overlap. C'est ce que recommandent les tutos. C'était mauvais.

Le problème : mes documents techniques ont des paragraphes denses. À 512 tokens, on coupe au milieu d'une explication. Le LLM reçoit un bout de phrase, ne comprend pas le contexte, et hallucine pour combler.

J'ai testé systématiquement 256, 512, 768 et 1024 tokens sur un jeu de 50 questions-réponses manuelles :

| Chunk size | Recall@5 | Faithfulness | Taux d'hallucination |
|-----------|----------|-------------|-------------------|
| 256       | 0.72     | 0.68        | 23%               |
| 512       | 0.78     | 0.74        | 18%               |
| **768**   | **0.83** | **0.81**    | **11%**           |
| 1024      | 0.80     | 0.79        | 13%               |

768 tokens était le sweet spot : assez long pour conserver le contexte, assez court pour que le retrieval reste précis. L'overlap de 128 tokens (17 %) garantit qu'aucune phrase importante ne tombe entre deux chunks.

Leçon : ne faites pas confiance aux valeurs par défaut. Testez sur vos données avec vos questions.

## 4. Le reranker : l'arme secrète sous-estimée

Sans reranker, le pipeline retournait des résultats « proches mais pas pertinents ». La similarité cosinus est bonne pour le recall, mauvaise pour la précision.

Exemple concret : pour la question « Quelle est l'architecture de MIRROR ? », le retrieval dense retournait :
1. Un chunk sur l'architecture de MIRROR (pertinent)
2. Un chunk sur l'architecture des Transformers (non pertinent : similarité cosinus haute car « architecture » est un terme dominant)
3. Un chunk sur les choix d'infrastructure de MIRROR (pertinent)

Le cross-encoder `ms-marco-MiniLM-L-6-v2` résout ça en ~80 ms. Il prend la question + chaque chunk et produit un score de pertinence contextuel, pas juste une distance vectorielle.

Mon setup : retrieve top-12 (filet large), puis le reranker garde le top-5 avec un seuil à -3,0 (tout ce qui est en dessous est du bruit). Le passage de top-3 à top-5 a été rendu possible par l'augmentation de la fenêtre de contexte à 8192 tokens.

Le piège du reranker : il charge un modèle BERT de ~80 Mo en RAM et ajoute ~80 ms de latence. Sur des requêtes simples (salutations, questions personnelles), c'est du gaspillage. D'où le query router.

## 5. Le Query Router : ne pas tout traiter pareil

Le query router classe les requêtes en trois tiers sans appel LLM, juste des regex et des heuristiques :

- **SIMPLE** (salutations, questions personnelles) : LLM direct, pas de RAG, ~100 ms
- **MEDIUM** (questions factuelles) : RAG standard, ~2-5 s
- **COMPLEX** (comparaisons, multi-parties, technique profond) : RAG étendu avec plus de chunks, ~5-15 s

Le routage prend <1 ms. Sur un mix de requêtes réaliste, ça réduit la latence moyenne de ~40 % parce que la majorité des interactions sont des salutations ou des questions simples.

Le code fait ~150 lignes de Python. Pas de ML, pas de classifieur entraîné. Des listes de mots-clés, des regex, et de la logique conditionnelle. Parfois la solution la plus simple est la meilleure.

## 6. Le prompt : l'itération invisible

Le prompt RAG a traversé une dizaine de versions. Chaque instruction corrige un bug observé :

- Sans « Read ALL documents », le modèle lisait le premier chunk et ignorait les autres
- Sans « Answer in the SAME language », il répondait en anglais à une question en français
- Sans « do NOT hallucinate », il inventait des réponses plausibles mais fausses
- Sans instruction sur les sources, il paraphrasait sans citer, rendant la vérification impossible

Un bon prompt RAG est un document de requirements. Chaque ligne corrige un bug observé en production.

## 7. Le context assembly

Assembler le contexte pour le LLM, ce n'est pas juste « concaténer les chunks » :

- **Tags de pertinence** : chaque chunk est annoté `[Relevance: HIGH/MEDIUM/LOW]` avec le score numérique. Le LLM peut pondérer.
- **Budget tokens dynamique** : le contexte est tronqué pour respecter `n_ctx - max_tokens - prompt_overhead`. Pas de troncature aveugle, on coupe au dernier chunk complet.
- **Ordre par pertinence** : les chunks les plus pertinents en premier. Les LLM ont un biais de position (primacy effect), mettez l'important au début.

## 8. L'infrastructure : de zéro à la production

C'est la partie que personne ne documente. Voici comment j'ai monté MIRROR de A à Z sur un seul serveur.

### 8.1 Le serveur

Hetzner AX102 dédié : 12 cœurs, 64 Go RAM DDR4, 2 × 512 Go NVMe. Coût : ~65 €/mois. Pas de GPU. Le choix est délibéré : prouver qu'un RAG complet peut tourner sur du hardware modeste.

Pourquoi un serveur dédié et pas du cloud (AWS/GCP) ? Parce que pour un projet personnel avec de l'inférence LLM qui tourne en continu, le dédié est 3-5× moins cher que des instances cloud équivalentes. Pas d'auto-scaling nécessaire, pas de latence réseau entre services.

### 8.2 Le nom de domaine et le DNS

Configuration DNS :
- Enregistrement A pointant vers l'IP du serveur
- Enregistrement AAAA si IPv6 disponible
- TTL de 300 s pour pouvoir migrer rapidement

### 8.3 TLS et reverse proxy avec Caddy

Caddy est le choix idéal pour un serveur solo :

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

Caddy gère automatiquement :
- Certificat TLS via Let's Encrypt (zéro configuration)
- Renouvellement automatique tous les 60 jours
- HTTP/2 et HTTP/3 par défaut
- Redirection HTTP vers HTTPS

Le `flush_interval -1` est critique pour le streaming SSE (Server-Sent Events). Sans ça, Caddy bufferise les réponses et le streaming ne fonctionne pas.

### 8.4 Docker Compose : tout orchestrer

Trois services dans un seul `docker-compose.yml` :

```yaml
services:
  caddy:        # Reverse proxy + TLS
  qdrant:       # Vector DB
  flask-app:    # Application (LLM + embedding + reranker)
```

Points critiques :
- `mlock: true` dans les options llama.cpp pour empêcher le swap du modèle
- Volume persistant pour Qdrant (les embeddings survivent aux redémarrages)
- Volume pour les certificats Caddy
- Health checks sur chaque service
- Restart policy `unless-stopped`

### 8.5 Monitoring en production

Métriques exposées via l'API :
- **embed_time** : temps d'embedding de la question (~30 ms)
- **search_time** : temps de recherche Qdrant (~15 ms)
- **rerank_time** : temps du cross-encoder (~80 ms)
- **generation_time** : temps de génération LLM (2-15 s selon complexité)
- **tokens/sec** : vitesse de génération en temps réel
- CPU par cœur, RAM, RSS du process, affichage live dans l'interface

## 9. Ce qui n'a PAS marché

### Semantic Query Cache
Cacher les réponses par similarité d'embedding. En théorie, gain de latence massif. En pratique, avec l'historique qui change le contexte, les hits étaient rares et les faux positifs dangereux.

### Chunking sémantique par embeddings
Couper quand la similarité cosinus entre fenêtres adjacentes chute. Trop lent à l'ingestion, résultats pas significativement meilleurs que le chunking fixe à 768 tokens.

### Query expansion avec LLM
Reformuler la question avec un appel LLM avant le retrieval. +5 % de recall, mais +2-3 s de latence. Sur CPU, chaque seconde compte. Le reranker apporte un meilleur gain pour moins de coût.

## Conclusion

Un RAG de production, ce n'est pas un pipeline linéaire qu'on branche et qui marche. C'est un système avec des dizaines de paramètres interdépendants qu'il faut calibrer sur ses propres données. Et c'est aussi une stack infra complète : serveur, DNS, TLS, reverse proxy, containers, monitoring.

Trois choses que j'aurais aimé savoir avant de commencer :
1. **Le chunking est le paramètre le plus impactant** : plus que le modèle d'embedding, plus que le LLM
2. **Le reranker vaut chaque milliseconde** : c'est le meilleur rapport qualité/coût du pipeline
3. **L'infrastructure, c'est 50 % du travail** : un bon RAG sur une mauvaise infra ne sert à rien

*Michail Berjaoui - Janvier 2025*
