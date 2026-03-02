---
title: "DevOps for ML Projects: Docker, Kubernetes, GitHub Actions and CI/CD Pipelines in Production"
date: 2025-06-10
tags: DevOps, Docker, Kubernetes, CI/CD, GitHub Actions, Pipeline, Production, MLOps, Docker Compose, Helm, ArgoCD
summary: "A comprehensive, pragmatic guide to setting up a modern DevOps pipeline for ML/AI projects. From containerization with Docker to Kubernetes orchestration, through GitHub Actions, deployment strategies, and monitoring - everything you need to go from a notebook to a production system. Based on real-world experience."
---

# DevOps for ML Projects: Docker, Kubernetes, GitHub Actions and CI/CD Pipelines in Production

> *"Your model works locally? Great. Now make it run in production, at 3 AM on a Sunday, when nobody's around to restart it."*

---

## Introduction

Let's be honest: most ML projects die between the Jupyter notebook and production. Not because the model is bad, but because nobody thought about the infrastructure. The data scientist delivers a `.py` file that "works on their machine", and then it's all downhill from there.

This article is the guide I wish I'd had when I had to set up my first real ML deployment pipeline. We'll cover the entire journey: from the `Dockerfile` to the Kubernetes cluster, through GitHub Actions, rollback strategies, and the small details that make the difference between a deployment that holds and one that blows up at 3 AM.

**What this article covers:**
1. Containerization with Docker (and ML-specific pitfalls)
2. Docker Compose for local development and staging
3. CI/CD pipelines with GitHub Actions
4. Orchestration with Kubernetes (Deployments, Services, ConfigMaps)
5. Helm Charts and GitOps with ArgoCD
6. Deployment strategies (Blue-Green, Canary, Rolling)
7. Monitoring and observability in production
8. Classic mistakes (and how to avoid them)

---

## 1. Docker: The Foundation of Everything

### Why Docker for ML?

If there's only one thing to remember: **Docker eliminates "it works on my machine"**. In ML, this is even more critical than in traditional development because dependencies are a nightmare:
- PyTorch 2.x with CUDA 12.1 but not 12.4
- `numpy` silently breaking between minor versions
- Compiled C++ libraries that refuse to work on a different distribution

### Anatomy of a Well-Built ML Dockerfile

```dockerfile
# === Stage 1: Builder ===
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy ONLY dependency files first (Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# === Stage 2: Runtime ===
FROM python:3.11-slim AS runtime

# Create non-root user (security)
RUN useradd --create-home appuser
WORKDIR /home/appuser/app

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=appuser:appuser . .

# Built-in healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser
EXPOSE 8000

CMD ["gunicorn", "app:create_app()", "-b", "0.0.0.0:8000", "-w", "4", "-k", "gthread"]
```

**Key points:**
- **Multi-stage build**: the builder is 1.2 GB, the runtime is 400 MB. In ML with PyTorch, the difference can be 8 GB vs 2 GB.
- **COPY order**: dependencies first, code second. Since dependencies rarely change, Docker uses the cache and rebuilds in 5 seconds instead of 5 minutes.
- **Non-root user**: NEVER run a container as root in production. That's the basics.
- **Healthcheck**: Kubernetes needs it to know if your pod is alive.

### The .dockerignore (don't forget it)

```
.git
__pycache__
*.pyc
.env
.venv
models/*.bin
models/*.gguf
*.egg-info
.pytest_cache
node_modules
```

Without `.dockerignore`, your build context includes your 4 GB models and your 500 MB `.git` folder. The build goes from 30 seconds to 10 minutes.

---

## 2. Docker Compose: Local and Staging Environment

Docker Compose is where you assemble the pieces. One file, and everyone on the team has exactly the same environment.

### docker-compose.yml for a typical ML project

```yaml
version: "3.9"

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
     - "8000:8000"
    environment:
     - FLASK_ENV=development
     - MODEL_PATH=/models/current
     - LOG_LEVEL=INFO
     - DATABASE_URL=postgresql://user:pass@db:5432/appdb
    volumes:
     - ./app:/home/appuser/app/app  # Hot-reload in dev
     - model-cache:/models
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: appdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
     - pgdata:/var/lib/postgresql/data
     - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d appdb"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

  nginx:
    image: nginx:alpine
    ports:
     - "80:80"
     - "443:443"
    volumes:
     - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
     - ./nginx/certs:/etc/nginx/certs:ro
    depends_on:
     - app

volumes:
  pgdata:
  model-cache:
```

**File breakdown:**

- **`app`**: the main service. The volume `./app:/home/appuser/app/app` mounts the local source code into the container for hot-reload during development (edit code, app restarts automatically). `depends_on` with `condition: service_healthy` ensures the database is ready before launching the app - without this, the app crashes on startup trying to connect to a DB that doesn't exist yet. `deploy.resources.limits` prevents a memory-hungry ML model from consuming all the dev machine's RAM.
- **`db`**: PostgreSQL with a built-in healthcheck. The `init.sql` file mounted in `docker-entrypoint-initdb.d/` is automatically executed on first startup to create tables. The named volume `pgdata` persists data across restarts.
- **`redis`**: in-memory cache with an LRU (Least Recently Used) eviction policy. When the 256 MB limit is reached, Redis removes the least-used keys. Useful for caching embeddings or frequent inference results.
- **`nginx`**: reverse proxy in front of the app. In production, it handles TLS termination, load balancing, and serves static files. The `:ro` (read-only) flag on volumes is a security best practice.
- **`volumes`** at the bottom: named volumes (`pgdata`, `model-cache`) persist data independently of container lifecycle. A `docker compose down` doesn't delete them (you need `docker compose down -v` for that).

### Docker Compose Profiles (the underrated feature)

```yaml
services:
  monitoring:
    image: grafana/grafana:latest
    profiles: ["monitoring"]
    ports:
     - "3000:3000"

  test-runner:
    build:
      context: .
      dockerfile: Dockerfile.test
    profiles: ["test"]
    command: pytest -v
```

```bash
# Normal dev
docker compose up

# Dev + monitoring
docker compose --profile monitoring up

# Run tests
docker compose --profile test run test-runner
```

---

## 3. GitHub Actions: The CI/CD Pipeline

This is where the magic happens. Every push triggers an automated chain: tests, build, security scan, deployment.

### Complete CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # === Job 1: Tests ===
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: testdb
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
     - uses: actions/checkout@v4

     - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

     - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt

     - name: Lint
        run: |
          ruff check .
          ruff format --check .

     - name: Run tests
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/testdb
        run: pytest -v --cov=app --cov-report=xml

     - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  # === Job 2: Build & Push Docker Image ===
  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
     - uses: actions/checkout@v4

     - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

     - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

     - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=ref,event=branch
            type=semver,pattern={{version}}

     - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # === Job 3: Security Scan ===
  security:
    needs: build
    runs-on: ubuntu-latest
    steps:
     - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: "sarif"
          output: "trivy-results.sarif"
          severity: "CRITICAL,HIGH"
```

**Pipeline breakdown:**

The workflow triggers on every push to `main` or `develop` and on pull requests to `main`. It chains three sequential jobs:

- **Job 1 - Tests**: spins up a PostgreSQL container as an auxiliary service (for integration tests), installs Python dependencies, runs linting with `ruff` (formatting + style), then runs tests with `pytest` and generates a coverage report sent to Codecov. If any test fails, the pipeline stops and the build never happens.
- **Job 2 - Build & Push**: only runs if tests pass (`needs: test`). Configures Docker Buildx for multi-platform builds, authenticates to GitHub Container Registry (GHCR) with the workflow's automatic token, then builds and pushes the Docker image. The `metadata-action` automatically generates tags (commit SHA, branch name, semantic version). GitHub Actions cache (`cache-from/to: type=gha`) speeds up subsequent builds by reusing unchanged Docker layers.
- **Job 3 - Security**: scans the Docker image with Trivy (Aqua Security's vulnerability scanner) to detect critical and high CVEs in dependencies. The result is exported in SARIF format, directly integrable into GitHub's Security tab.

### CD Pipeline (deployment)

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  workflow_run:
    workflows: ["CI Pipeline"]
    types: [completed]
    branches: [main]

jobs:
  deploy-staging:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    environment: staging

    steps:
     - uses: actions/checkout@v4

     - name: Set up kubectl
        uses: azure/setup-kubectl@v3

     - name: Configure kubeconfig
        run: echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > $HOME/.kube/config

     - name: Deploy to staging
        run: |
          kubectl set image deployment/app \
            app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n staging
          kubectl rollout status deployment/app -n staging --timeout=300s

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://app.example.com

    steps:
     - uses: actions/checkout@v4

     - name: Configure kubeconfig
        run: echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > $HOME/.kube/config

     - name: Deploy to production (canary)
        run: |
          # Deploy canary (10% of traffic)
          kubectl apply -f k8s/canary-deployment.yaml -n production
          sleep 60

          # Check canary metrics
          ERROR_RATE=$(kubectl exec -n monitoring deploy/prometheus -- \
            promtool query instant 'rate(http_requests_total{status=~"5.."}[5m])')

          if [ $(echo "$ERROR_RATE > 0.05" | bc) -eq 1 ]; then
            echo "Canary failed - rolling back"
            kubectl rollout undo deployment/app-canary -n production
            exit 1
          fi

          # Promote canary to full production
          kubectl set image deployment/app \
            app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n production
          kubectl rollout status deployment/app -n production --timeout=300s
```

**Deployment breakdown:**

The CD workflow triggers automatically when the CI pipeline succeeds on `main` (`workflow_run`). It chains two environments:

- **Staging**: `kubectl set image` updates the Deployment's image with the commit SHA (unique and immutable tag). `kubectl rollout status` blocks until all pods are updated or times out after 5 minutes. If the rollout fails, the job fails and production is never touched.
- **Production**: only runs if staging succeeded. Uses a canary strategy: first deploy the new version to a small subset, wait 60 seconds, then check the error rate via Prometheus. If the rate exceeds 5%, automatic rollback and stop. Otherwise, full rollout. The `environment: production` in GitHub allows adding manual approval rules (a human must validate before deployment).

### Secrets and Security

**NEVER** put secrets in your code. Use:
- **GitHub Secrets** for tokens, passwords, API keys
- **GitHub Environments** to separate staging/production with approval rules
- **Sealed Secrets** or **External Secrets Operator** on the Kubernetes side

---

## 4. Kubernetes: Production Orchestration

### Why Kubernetes?

Docker Compose is fine for dev and small deployments. But when you need:
- **Auto-scaling**: scale to 10 replicas when traffic spikes
- **Self-healing**: automatically restart a crashing pod
- **Zero-downtime deployments**: deploy without cutting the service
- **Multi-environment**: staging, production, with network isolation

That's when Kubernetes becomes indispensable.

### Essential Deployment Files

#### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  labels:
    app: ml-api
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Zero-downtime
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
        version: v1
    spec:
      containers:
       - name: app
          image: ghcr.io/myorg/ml-api:latest
          ports:
           - containerPort: 8000
          env:
           - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: database-url
           - name: MODEL_PATH
              value: /models/current
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          volumeMounts:
           - name: model-storage
              mountPath: /models
      volumes:
       - name: model-storage
          persistentVolumeClaim:
            claimName: model-pvc
```

**Walking through the key sections:**

- **`replicas: 3`** - runs three copies of the app for redundancy. If one crashes, the other two keep serving traffic.
- **`strategy.rollingUpdate`** with `maxUnavailable: 0` - during a deploy, Kubernetes starts new pods *before* killing old ones. Users never see downtime.
- **`resources.requests`** vs **`limits`** - `requests` is what the scheduler guarantees (the pod gets at least 500m CPU and 1 GB RAM). `limits` is the ceiling - if the pod tries to use more than 4 GB, it gets OOM-killed. For ML workloads, set limits generously because model loading can spike memory.
- **`livenessProbe`** - Kubernetes pings `/health` every 10s. If it fails 3 times, the pod is killed and restarted. The `initialDelaySeconds: 30` gives the app time to load models before checks begin.
- **`readinessProbe`** - similar, but controls whether the pod receives traffic. A pod that's alive but not ready (still loading a model) won't get requests routed to it.
- **`volumeMounts` + PVC** - the model files live on a `PersistentVolumeClaim`, so they survive pod restarts and don't need to be re-downloaded each time.

#### Service & Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: ml-api
  ports:
   - port: 80
      targetPort: 8000
  type: ClusterIP
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
   - hosts:
       - api.example.com
      secretName: api-tls
  rules:
   - host: api.example.com
      http:
        paths:
         - path: /
            pathType: Prefix
            backend:
              service:
                name: app-service
                port:
                  number: 80
```

**What this does:**

- The **Service** is an internal load balancer. It routes traffic to any pod matching the `app: ml-api` label. `ClusterIP` means it's only reachable inside the cluster - external traffic comes through the Ingress.
- The **Ingress** is the external entry point. It terminates TLS (via `cert-manager`, which auto-provisions Let's Encrypt certificates), enforces rate limiting (100 req/min via nginx annotations), and routes HTTPS traffic to the internal Service on port 80.
- The separation matters: you can swap the Ingress controller (nginx → Traefik → Envoy) without touching your app. The Service abstraction means pods can scale up/down and traffic is automatically balanced.

#### HorizontalPodAutoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app
  minReplicas: 2
  maxReplicas: 10
  metrics:
   - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
   - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
       - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
       - type: Pods
          value: 1
          periodSeconds: 120
```

**How auto-scaling works:**

- `minReplicas: 2` - always at least 2 pods running (redundancy baseline).
- When average CPU across all pods exceeds 70%, Kubernetes adds pods (up to 10). When it drops, it removes them.
- **`stabilizationWindowSeconds`** prevents flapping: scale-up waits 60s of sustained high load before adding pods, scale-down waits 300s (5 min) before removing. This avoids the "scale up, traffic spike ends, scale down, next spike, scale up again" loop.
- The `policies` limit the rate of change: max 2 pods added per minute, max 1 removed per 2 minutes. Aggressive scale-up, conservative scale-down - this is the safe default for ML workloads where cold starts are expensive.

### ConfigMaps and Secrets

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: "INFO"
  MAX_CONTEXT_LENGTH: "4096"
  CACHE_TTL: "3600"
  RATE_LIMIT: "100"
---
# k8s/secret.yaml (in prod, use Sealed Secrets)
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
stringData:
  database-url: "postgresql://user:pass@db:5432/appdb"
  api-key: "sk-..."
```

**The difference between ConfigMap and Secret:**

- **ConfigMap**: non-sensitive configuration, stored in plaintext in etcd. Ideal for application parameters (log level, timeouts, limits). Modifiable without rebuilding the image - a `kubectl rollout restart` is enough for pods to reload the config.
- **Secret**: sensitive data (passwords, API keys, certificates). Base64-encoded in etcd (not encrypted by default - you need to enable at-rest encryption on the cluster). In production, use **Sealed Secrets** (asymmetric encryption, encrypted secrets can live in Git) or **External Secrets Operator** (syncs from Vault, AWS Secrets Manager, etc.).
- Both are injected the same way into pods: via `envFrom` (all keys become environment variables) or via `valueFrom.secretKeyRef` (key by key, as in the Deployment above).

---

## 5. Helm & GitOps: The Next Level

### Helm: Templating Kubernetes

When you have 15 nearly identical YAML files for staging and production, Helm saves your life.

```yaml
# helm/values-staging.yaml
replicaCount: 2
image:
  repository: ghcr.io/myorg/ml-api
  tag: develop
resources:
  limits:
    memory: 2Gi
    cpu: 1000m
ingress:
  host: staging-api.example.com

# helm/values-production.yaml
replicaCount: 5
image:
  repository: ghcr.io/myorg/ml-api
  tag: v1.2.3
resources:
  limits:
    memory: 4Gi
    cpu: 2000m
ingress:
  host: api.example.com
```

**How it works:**

A single Kubernetes template (with variables like `{{ .Values.replicaCount }}`, `{{ .Values.image.tag }}`, etc.) and per-environment value files. Staging runs with 2 replicas, a develop branch image, and 2 GB of RAM. Production runs with 5 replicas, a stable tagged version (`v1.2.3`), and 4 GB of RAM. A single `helm upgrade --values values-production.yaml` applies all the differences at once, without duplicating YAML files.

### GitOps with ArgoCD

The principle: **Git is the source of truth**. You never run `kubectl apply` manually. You push to Git, ArgoCD detects the change and syncs the cluster.

```yaml
# argocd/application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-api-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/ml-api-infra
    targetRevision: main
    path: helm
    helm:
      valueFiles:
       - values-production.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
     - CreateNamespace=true
```

**Benefits:**
- Complete audit trail (every deployment = a commit)
- Rollback = `git revert`
- Total reproducibility
- Drift detection (ArgoCD corrects manual changes)

---

## 6. Deployment Strategies

### Rolling Update (K8s default)

Progressively replaces old pods with new ones. Simple, reliable, sufficient in 80% of cases.

```
[v1] [v1] [v1]  -->  [v2] [v1] [v1]  -->  [v2] [v2] [v1]  -->  [v2] [v2] [v2]
```

### Blue-Green

Two identical environments. Traffic switches all at once from old (blue) to new (green). Instant rollback: switch back to blue.

```
Blue  [v1] [v1] [v1]  <-- traffic
Green [v2] [v2] [v2]      (ready)

# Switch
Blue  [v1] [v1] [v1]      (standby)
Green [v2] [v2] [v2]  <-- traffic
```

### Canary

Deploy the new version to a small percentage of traffic (5-10%), monitor metrics, then gradually increase.

```
[v1] [v1] [v1] [v1] [v1] [v1] [v1] [v1] [v1] [v2]  <-- 10% canary
                            ...monitoring...
[v2] [v2] [v2] [v2] [v2] [v2] [v2] [v2] [v2] [v2]  <-- 100% rollout
```

**My advice**: start with Rolling Update. Move to Canary when you have monitoring in place and know which metrics to watch. Blue-Green if you need instant rollback and can afford to double the resources.

---

## 7. Monitoring and Observability

A deployment without monitoring is like driving at night without headlights. You're moving forward, but you can't see the wall.

### The Standard Stack

```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
     - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
     - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    environment:
     - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
     - ./monitoring/dashboards:/var/lib/grafana/dashboards
    ports:
     - "3000:3000"

  loki:
    image: grafana/loki:latest
    ports:
     - "3100:3100"
```

**The three pillars of observability:**

- **Prometheus**: collects metrics. It scrapes (queries) the `/metrics` endpoints of each service at regular intervals. Metrics are stored locally as time series. The `prometheus.yml` file defines which services to scrape and how often.
- **Grafana**: visualization. Connects to Prometheus as a data source and displays metrics as dashboards. Pre-configured dashboards in `./monitoring/dashboards` are loaded automatically on startup.
- **Loki**: log aggregation. It's the "Prometheus for logs" - same query interface (LogQL), same Grafana integration. Applications send their logs to Loki via Promtail or directly, and you can correlate them with metrics in Grafana.

### Essential Metrics for ML in Production

| Metric | Alert Threshold | Why |
|:---|:---|:---|
| P99 Latency | > 2s | Degraded UX, client timeouts |
| 5xx Error Rate | > 1% | Application problem |
| CPU utilization | > 80% sustained | Need to scale up |
| Memory utilization | > 85% | OOM kill risk |
| Inference time | > 5s (ML specific) | Model too slow, optimize |
| Queue depth | > 100 requests | Backpressure, scale out |
| Pod restart count | > 3/hour | Crash loop, investigate |

### Alerting with Prometheus

```yaml
# monitoring/alerts.yml
groups:
 - name: app-alerts
    rules:
     - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected (> 5%)"

     - alert: HighLatency
        expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency > 2s"
```

**How to read these rules:**

- **`HighErrorRate`**: `rate(http_requests_total{status=~"5.."}[5m])` computes the rate of requests with a 5xx HTTP status over the last 5 minutes. If this rate exceeds 5% (`> 0.05`) for 2 consecutive minutes (`for: 2m`), the alert fires as `critical`. The `for` clause prevents false positives from an isolated spike.
- **`HighLatency`**: `histogram_quantile(0.99, ...)` computes the 99th percentile of latency - meaning 99% of requests are faster than this value. If p99 exceeds 2 seconds for 5 minutes, a `warning` alert fires. We use p99 instead of average because the average hides outliers: an average of 200ms can mask 1% of requests at 10s.

These alerts are sent via Alertmanager to Slack, PagerDuty, or email based on the `severity` label.

---

## 8. Classic Mistakes (and How to Avoid Them)

After a few years in the game, here are the mistakes I see most often:

### 1. No healthcheck
Kubernetes doesn't know if your app is alive. It keeps routing traffic to a dead pod. **Always** implement `/health` and `/ready`.

### 2. 8 GB Docker images
Use multi-stage builds. Separate build dependencies from runtime dependencies. Use `-slim` or `-alpine` images.

### 3. Hardcoded secrets
We still see this in 2025. Use environment variables, GitHub Secrets, Vault, or Sealed Secrets. Period.

### 4. No resource limits
Without limits, a single pod can consume all the node's memory and crash the others. ALWAYS set `requests` and `limits`.

### 5. Deploying on Friday
I'm kidding. Or am I? If your CI/CD pipeline is solid and you have canary in place, you can deploy anytime. That's precisely the point of everything we've set up.

### 6. Ignoring logs
Centralize your logs with Loki, ELK, or CloudWatch. `print()` statements in the console of a pod that's going to be recycled in 2 minutes doesn't count as logging.

### 7. No rollback plan
If you can't rollback in 30 seconds, you're not ready for production. Kubernetes rollback, Git revert + ArgoCD sync, or Blue-Green switch.

---

## Conclusion

DevOps for ML isn't sexy. It's YAML, pipelines, healthchecks, and a lot of debugging Docker permissions at 11 PM. But it's what makes the difference between a side project and a product.

The typical journey:

1. **Week 1**: Clean Dockerfile + Docker Compose for local dev
2. **Week 2**: GitHub Actions CI (tests + build + push image)
3. **Week 3**: Kubernetes deployment (Deployment + Service + Ingress)
4. **Week 4**: Monitoring (Prometheus + Grafana) + alerts
5. **Month 2**: Helm + ArgoCD + Canary deployments

You don't need to do everything at once. But each step gets you closer to a system that runs on its own - even at 3 AM on a Sunday.

---

## References

1. Docker Documentation. *Best practices for writing Dockerfiles*. docs.docker.com
2. Kubernetes Documentation. *Deployments*. kubernetes.io/docs/concepts/workloads/controllers/deployment
3. GitHub Actions Documentation. *Building and testing Python*. docs.github.com/en/actions
4. Helm Documentation. *Getting Started*. helm.sh/docs
5. ArgoCD Documentation. *Getting Started*. argo-cd.readthedocs.io
6. Prometheus Documentation. *Alerting Rules*. prometheus.io/docs
7. Google SRE Book (2016). *Site Reliability Engineering*. sre.google/sre-book
8. Sculley et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NIPS 2015
9. Burns, B. et al. (2016). *Borg, Omega, and Kubernetes*. ACM Queue

---

*Michail Berjaoui - June 2025*
