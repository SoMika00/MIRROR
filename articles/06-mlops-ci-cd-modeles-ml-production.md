---
title: "Deployer un LLM en Production : FastAPI, Docker et Kubernetes"
date: 2024-08-22
tags: FastAPI, Docker, Kubernetes, Production, LLM, Deployment, Infrastructure
summary: "Guide complet pour déployer un service LLM en production : API FastAPI avec streaming, containerisation Docker multi-stage, orchestration Kubernetes avec GPU scheduling, et bonnes pratiques de monitoring."
---

# Deployer un LLM en Production : FastAPI, Docker et Kubernetes

## Introduction

Construire un prototype LLM en local est la partie facile. Le vrai défi commence quand il faut le rendre accessible à des utilisateurs réels : haute disponibilité, scaling, monitoring, sécurité, et zero-downtime deployments. Cet article couvre l'architecture complète pour déployer un service LLM en production, du code Python jusqu'au cluster Kubernetes.

## 1. API FastAPI : le Point d'Entrée

### Pourquoi FastAPI ?

FastAPI est le framework Python dominant pour les APIs ML en production :
- **Async natif** : gère des centaines de requêtes concurrentes (essentiel quand le LLM est lent)
- **Pydantic** : validation automatique des entrées/sorties, documentation OpenAPI générée
- **Streaming SSE** : streaming token-par-token pour une UX responsive
- **Performances** : comparable à Go/Node.js grâce à Uvicorn (ASGI)

### Structure d'une API LLM

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="LLM Service")

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.post("/v1/chat")
async def chat(req: ChatRequest):
    # Non-streaming response
    response = llm.generate(req.message, max_tokens=req.max_tokens)
    return {"response": response, "tokens_used": len(response)}

@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    # SSE streaming for real-time token display
    async def generate():
        for token in llm.stream(req.message):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": llm.is_loaded()}
```

### Bonnes pratiques

- **Health checks** : `/health` et `/ready` pour les probes Kubernetes (liveness et readiness)
- **Timeouts** : configurer des timeouts par requête (ex: 120s pour la génération)
- **Graceful shutdown** : intercepter SIGTERM pour finir les générations en cours avant l'arrêt
- **Rate limiting** : `slowapi` ou middleware custom pour éviter les abus
- **Structured logging** : JSON logs avec request_id, latency, tokens_used pour l'observabilité
- **CORS** : configurer correctement pour les frontends web

## 2. Docker : Containerisation

### Dockerfile Multi-Stage

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y build-essential cmake
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime (slim)
FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY ./app /app
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Bonnes pratiques Docker pour ML

- **Multi-stage builds** : séparer la compilation (cmake, build-essential) du runtime pour réduire la taille de l'image
- **Modèles hors de l'image** : monter les poids en volume (`-v ./models:/models`), ne pas les inclure dans l'image Docker
- **Pin des versions** : `requirements.txt` avec versions exactes (`llama-cpp-python==0.3.4`)
- **Non-root user** : `USER 1000` pour la sécurité
- **NVIDIA Container Toolkit** : pour le GPU passthrough (`--gpus all`)

### Docker Compose pour le dev et le single-server

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    volumes:
      - ./models:/models
    environment:
      - MODEL_PATH=/models/phi-4-Q4_K_M.gguf
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    depends_on:
      qdrant:
        condition: service_healthy

  qdrant:
    image: qdrant/qdrant:v1.12.4
    volumes: ["./qdrant_data:/qdrant/storage"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]

  caddy:
    image: caddy:2.8
    ports: ["443:443", "80:80"]
    volumes: ["./Caddyfile:/etc/caddy/Caddyfile"]
```

## 3. Kubernetes : Orchestration à l'Echelle

### Quand passer de Compose à Kubernetes ?

- Vous avez besoin de **scaling horizontal** (plusieurs replicas de l'API)
- Vous servez **plusieurs modèles** avec du GPU scheduling
- Vous avez besoin de **rolling updates** (zero-downtime deployments)
- Votre infrastructure est **multi-node** (plusieurs serveurs)

### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: api
        image: registry.example.com/llm-api:v1.2.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "20Gi"
            nvidia.com/gpu: "1"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### GPU Scheduling

Pour les workloads LLM, le GPU scheduling est critique :
- Installer le **NVIDIA device plugin** DaemonSet
- Créer des **node pools** GPU séparés avec taints/tolerations
- Utiliser `resources.limits: nvidia.com/gpu: 1` pour réserver un GPU par pod
- Considérer le **MIG (Multi-Instance GPU)** sur A100/H100 pour partager un GPU entre plusieurs pods

## 4. Monitoring et Observabilité

### Métriques essentielles

| Métrique | Outil | Seuil d'alerte |
|:---|:---|:---|
| Time to First Token (TTFT) | Prometheus + custom metric | p95 > 500ms |
| Total generation time | Prometheus | p95 > 30s |
| Tokens per second | Custom metric | < 5 t/s (CPU) |
| GPU memory utilization | DCGM exporter | > 90% |
| Error rate (5xx) | Prometheus | > 1% |
| Request queue depth | Custom metric | > 50 |

### Stack recommandée

- **Prometheus + Grafana** : métriques et dashboards
- **Loki** : aggregation de logs
- **Langfuse** : tracing LLM-spécifique (prompt, context, response, latency, tokens)
- **Alertmanager** : alertes vers Slack/PagerDuty

## 5. Sécurité en Production

- **API keys** : authentification par token (header `Authorization: Bearer xxx`)
- **Rate limiting** : par IP et par clé API
- **Input sanitization** : limiter la taille des prompts, filtrer les injections
- **Output scanning** : détecter les PII (noms, emails, numéros) dans les réponses
- **Network policies** : le LLM ne doit pas avoir accès à internet (exfiltration)
- **Secrets management** : Kubernetes Secrets ou HashiCorp Vault, jamais en clair dans les manifests

## 6. CI/CD Pipeline

```
1. [Push to main]     -> Build Docker image (GitHub Actions / GitLab CI)
2. [Run tests]        -> Unit tests + regression tests (50 golden Q&A pairs)
3. [Build & Push]     -> Push image to container registry (GHCR, ECR)
4. [Deploy staging]   -> Apply K8s manifests to staging namespace
5. [Smoke tests]      -> Health check + sample generation test
6. [Deploy prod]      -> Rolling update (maxSurge: 1, maxUnavailable: 0)
7. [Monitor]          -> Watch error rate and latency for 30min post-deploy
```

## Conclusion

Déployer un LLM en production nécessite une stack complète : FastAPI pour l'API, Docker pour la portabilité, Kubernetes pour le scaling et la résilience. Les clés du succes : health checks robustes, monitoring dès le jour 1, et rolling updates pour le zero-downtime. Commencez avec Docker Compose pour valider votre architecture, puis migrez vers Kubernetes quand le scaling devient nécessaire.
