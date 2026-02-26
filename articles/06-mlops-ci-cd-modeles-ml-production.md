---
title: "Déployer un LLM sur un Seul Serveur : Le Guide Anti-Kubernetes"
date: 2024-08-22
tags: Docker, Caddy, Production, LLM, Deployment, Infrastructure, CPU, llama.cpp, mlock
summary: "Retour d'expérience sur le déploiement de MIRROR : un service LLM complet (RAG + multi-modèles + monitoring temps réel) sur un seul serveur Hetzner à 64 Go RAM, sans GPU, sans Kubernetes, sans cloud provider. Docker Compose, Caddy, et des choix d'architecture opiniâtres."
---

# Déployer un LLM sur un Seul Serveur : Le Guide Anti-Kubernetes

## L'opinion impopulaire

Kubernetes est un outil formidable. Mais si vous déployez un seul service LLM pour un portfolio, un side project, ou un MVP, **vous n'en avez pas besoin**. Et le fait que 90% des articles de déploiement LLM commencent par un manifeste K8s est un problème pour les gens qui veulent juste mettre quelque chose en ligne.

MIRROR tourne sur un seul serveur Hetzner AX102 (12 vCPU AMD, 64 Go RAM, 2×512 Go NVMe, zéro GPU) à ~65€/mois. Il sert un pipeline RAG complet avec switching dynamique entre 8 modèles LLM (3.8B à 42B paramètres), un modèle d'embedding, un reranker, une base vectorielle, et un reverse proxy HTTPS. Tout ça dans 3 containers Docker.

Cet article documente comment.

## 1. L'architecture : 3 containers et c'est tout

```yaml
services:
  caddy:        # Reverse proxy + HTTPS automatique
    image: caddy:2.8.4
    ports: ["80:80", "443:443"]

  qdrant:       # Base vectorielle
    image: qdrant/qdrant:v1.12.4
    expose: ["6333"]

  mirror:       # App Flask + LLM + Embedding + Reranker
    build: .
    expose: ["5000"]
    volumes:
      - ./models:/app/models    # GGUF poids montés, PAS dans l'image
```

Pourquoi pas FastAPI ? Parce que Flask suffit. MIRROR n'a pas besoin d'async natif — le LLM est le bottleneck, pas le framework web. La génération est verrouillée par un `threading.Lock()` de toute façon (un seul modèle en mémoire, un seul thread d'inférence à la fois). Flask + Gunicorn avec 1 worker suffit largement.

**Pourquoi Caddy et pas Nginx ?** Trois mots : HTTPS automatique. Caddy provisionne et renouvelle les certificats Let's Encrypt sans aucune configuration. Mon `Caddyfile` fait 12 lignes :

```
mymirror.fr {
    encode gzip zstd
    header {
        X-Content-Type-Options nosniff
        X-Frame-Options DENY
        Referrer-Policy strict-origin-when-cross-origin
    }
    reverse_proxy mirror:5000
}

www.mymirror.fr {
    redir https://mymirror.fr{uri} permanent
}
```

Comparé aux ~60 lignes de config Nginx + Certbot + cron de renouvellement, le choix est évident.

## 2. Le Dockerfile ML : chaque Mo compte

Le Dockerfile de MIRROR compile `llama-cpp-python` depuis les sources avec les optimisations AVX2/FMA :

```dockerfile
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y build-essential cmake
COPY requirements.txt .
ENV CMAKE_ARGS="-DGGML_AVX2=ON -DGGML_FMA=ON"
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . /app
WORKDIR /app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "1", "--timeout", "300", "run:app"]
```

### Les décisions qui comptent :

1. **Multi-stage build** : la couche `builder` avec `cmake` et `build-essential` fait ~800 Mo. Le runtime final fait ~1.2 Go sans les poids. Sans multi-stage, ce serait 2 Go+.

2. **Modèles en volume, JAMAIS dans l'image** : un GGUF de Phi-4 fait 2.5 Go, Qwen 32B fait 20 Go. Si vous mettez ça dans l'image Docker, chaque rebuild est une torture. Volume bind mount : `./models:/app/models`.

3. **1 worker Gunicorn** : contre-intuitif, mais correct. Le LLM monopolise toute la RAM et le CPU pendant la génération. Deux workers = deux copies du modèle en mémoire = OOM. Le secret : `--timeout 300` pour les longues générations.

4. **AVX2/FMA flags** : sur les CPU AMD Zen 3/4 de Hetzner, ces flags améliorent la vitesse d'inférence de ~30%. Ne les oubliez pas.

## 3. La gestion mémoire : le vrai boss final

Sur un serveur à 64 Go de RAM partagés entre Flask, le LLM, l'embedding, le reranker, Qdrant, et le système, chaque composant doit être mesuré :

| Composant | RAM (mesuré) |
|-----------|-------------|
| Système + Docker overhead | ~2 Go |
| Qdrant (avec index HNSW) | 2-4 Go |
| BGE-M3 (embedding) | ~2.5 Go |
| ms-marco-MiniLM reranker | ~0.08 Go |
| Flask + app code | ~0.3 Go |
| **Phi-4 Mini 3.8B Q4_K_M** | **~3 Go** |
| **Phi-4 14B Q4_K_M** | **~9 Go** |
| **Qwen 2.5 32B Q4_K_M** | **~20 Go** |
| Buffers/cache Linux | ~4 Go |

Total pour le setup par défaut (Phi-4 Mini) : ~12 Go. Ça laisse ~52 Go libres — assez pour charger Qwen 32B à la place si un utilisateur le demande.

### mlock : l'optimisation invisible

`mlock` empêche Linux de swapper les poids du modèle vers le disque. Sans mlock, le kernel peut décider de mettre des pages du modèle en swap quand Qdrant fait une grosse recherche. Résultat : la prochaine génération met 30 secondes au lieu de 5 parce qu'il faut re-lire les poids depuis le NVMe.

```python
self.model = Llama(
    model_path=path,
    n_ctx=8192,
    n_threads=12,      # tous les cœurs
    n_batch=1024,       # batch size pour le prompt processing
    use_mlock=True,     # épingler en RAM
    use_mmap=True,      # memory-mapped file pour le chargement initial
)
```

`mmap` + `mlock` = le kernel mappe le fichier GGUF en mémoire (rapide, pas de copie) puis le verrouille en RAM (pas de swap). C'est ~2x plus rapide au chargement qu'un `read()` classique et garantit une latence stable.

## 4. Le monitoring fait maison

Pas de Prometheus, pas de Grafana. Juste un endpoint `/api/models/metrics` qui lit `/proc/stat` et `/proc/meminfo` en temps réel :

```python
# CPU : différentiel /proc/stat entre deux polls (pas load average !)
def _cpu_percent_realtime():
    current = _read_proc_stat()
    d_total = current["total"] - prev["total"]
    d_idle = current["idle"] - prev["idle"]
    return ((d_total - d_idle) / d_total) * 100.0
```

Le frontend poll chaque seconde et affiche :
- **CPU agrégé** : pourcentage total
- **CPU par cœur** : mini barres verticales (on voit exactement quels cœurs sont occupés par l'inférence)
- **RAM** : total, utilisé, buffers/cache
- **Process RSS** : combien le process Flask consomme exactement
- **Inference telemetry** : tokens générés, tokens/seconde, en temps réel pendant la génération

Pourquoi pas load average pour le CPU ? Parce que le load average est une moyenne exponentielle sur 1/5/15 minutes. Quand un LLM sature les 12 cœurs pendant 5 secondes puis s'arrête, le load average ne reflète rien d'utile. Le différentiel `/proc/stat` donne le CPU **en ce moment**, ce qui est ce que l'utilisateur veut voir.

## 5. Le switching de modèles à chaud

MIRROR permet de changer de LLM à la volée via l'interface. L'utilisateur clique sur "Qwen 2.5 32B" dans le model manager, et :

1. Le modèle actuel est déchargé (`del self.model` + `gc.collect()`)
2. Le nouveau GGUF est chargé avec les paramètres adaptés (n_ctx, n_threads)
3. Le frontend affiche la progression

Le piège : `gc.collect()` ne suffit pas toujours. `llama-cpp-python` alloue de la mémoire via malloc côté C++. J'ai dû forcer un `del` explicite du modèle Python ET vérifier que le RSS redescend effectivement avant de charger le nouveau modèle.

Si le modèle n'est pas encore téléchargé, le système le pull depuis HuggingFace via `huggingface_hub`. Les modèles sont stockés dans `./models/` (volume Docker), donc ils persistent entre les redéploiements.

## 6. HTTPS et DNS : les 2 heures les plus frustrantes

C'est le genre de chose qui devrait prendre 5 minutes. Configurer le DNS, pointer vers le serveur, Caddy s'occupe du reste. En pratique :

- **Piège n°1** : votre registrar (Gandi, OVH, etc.) a souvent un **hébergement web par défaut** qui intercepte le trafic HTTP. Caddy ne peut pas passer le challenge ACME de Let's Encrypt parce que la requête de validation arrive chez le registrar au lieu de votre serveur. Solution : désactiver l'hébergement web et les redirections web chez le registrar.

- **Piège n°2** : le TTL DNS. Même après avoir changé les records, les anciens sont cached pendant le TTL (souvent 1800s = 30 min). Mettez le TTL à 300s avant de migrer.

- **Piège n°3** : Caddy cache ses échecs ACME. Si la première tentative échoue (DNS pas encore propagé), Caddy attend 6 heures avant de réessayer. Solution : `docker compose down && docker volume rm mirror_caddy_data && docker compose up -d` pour repartir à zéro.

## 7. Ce que Kubernetes apporterait (et ce qui me manque pas)

Pour être honnête sur les trade-offs :

### Ce qui me manque :
- **Zero-downtime redeploy** : quand je rebuild le container, il y a ~10s de downtime. Acceptable pour un portfolio, pas pour un SaaS.
- **Auto-restart sur OOM** : si le modèle 32B OOM, Docker le restart mais c'est pas gracieux.

### Ce dont je n'ai PAS besoin :
- **Horizontal scaling** : un seul utilisateur génère à la fois (lock). Plus de replicas ne changerait rien.
- **GPU scheduling** : pas de GPU.
- **Service mesh** : 3 containers sur le même réseau Docker. Pas besoin d'Istio.
- **Config maps et secrets** : un fichier `.env` et un `docker-compose.yml` suffisent.
- **Helm charts** : pour un seul service ? Non.

Le jour où MIRROR aura besoin de scaling horizontal (plusieurs utilisateurs simultanés avec des modèles différents), je migrerai. Pas avant.

## 8. Le script de déploiement : 40 lignes de bash

```bash
#!/bin/bash
set -e

echo "[1/4] Building..."
docker compose build --no-cache mirror

echo "[2/4] Starting..."
docker compose up -d

echo "[3/4] Health check..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:5000/api/chat/status > /dev/null; then
        echo "Ready!"
        break
    fi
    sleep 2
done

echo "[4/4] Downloading models if missing..."
docker compose exec mirror python -c "from app.config import MODEL_REGISTRY; print('OK')"

echo "Done. https://mymirror.fr"
```

Pas de CI/CD. `git pull && ./deploy.sh`. Pour un side project, c'est parfait. Quand j'aurai besoin de GitHub Actions, je les ajouterai. Mais pas avant que la complexité le justifie.

## Conclusion

Le déploiement LLM en production n'a pas besoin d'être compliqué. Docker Compose + Caddy + un serveur dédié suffisent pour un service fiable, sécurisé, et performant. Les clés :

1. **Un seul worker** pour le LLM — pas de concurrence sur la RAM
2. **mlock + mmap** pour une latence stable et un chargement rapide
3. **Volumes pour les modèles** — jamais dans l'image Docker
4. **Monitoring maison** via `/proc` — simple, précis, zéro dépendance
5. **Caddy pour HTTPS** — Let's Encrypt automatique, zéro maintenance

Ajoutez de la complexité quand le besoin se présente, pas avant. YAGNI s'applique aussi à l'infrastructure.
