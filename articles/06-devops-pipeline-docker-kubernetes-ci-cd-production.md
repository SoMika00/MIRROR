---
title: "DevOps pour Projets ML : Docker, Kubernetes, GitHub Actions et Pipelines CI/CD en Production"
date: 2025-06-10
tags: DevOps, Docker, Kubernetes, CI/CD, GitHub Actions, Pipeline, Production, MLOps, Docker Compose, Helm, ArgoCD
summary: "Un guide complet et pragmatique sur la mise en place d'un pipeline DevOps moderne pour des projets ML/IA. De la conteneurisation avec Docker à l'orchestration Kubernetes, en passant par les GitHub Actions, les stratégies de déploiement et le monitoring - tout ce qu'il faut pour passer d'un notebook à un système en production. Basé sur l'expérience terrain."
---

# DevOps pour Projets ML : Docker, Kubernetes, GitHub Actions et Pipelines CI/CD en Production

> *"Ton modèle marche en local ? Super. Maintenant fais-le tourner en prod, à 3h du matin, un dimanche, quand personne n'est là pour le relancer."*

---

## Introduction

On va se dire les choses : la majorité des projets ML meurent entre le notebook Jupyter et la production. Pas parce que le modèle est mauvais, mais parce que personne n'a pensé à l'infra. Le data scientist livre un `.py` qui "marche sur sa machine", et ensuite c'est la descente aux enfers.

Cet article est le guide que j'aurais voulu avoir quand j'ai dû mettre en place mon premier vrai pipeline de déploiement ML. On va couvrir tout le chemin : du `Dockerfile` jusqu'au cluster Kubernetes, en passant par les GitHub Actions, les stratégies de rollback, et les petits détails qui font la différence entre un déploiement qui tient et un qui explose à 3h du mat.

**Ce que cet article couvre :**
1. Conteneurisation avec Docker (et les pièges spécifiques au ML)
2. Docker Compose pour le développement local et le staging
3. Pipelines CI/CD avec GitHub Actions
4. Orchestration avec Kubernetes (Deployments, Services, ConfigMaps)
5. Helm Charts et GitOps avec ArgoCD
6. Stratégies de déploiement (Blue-Green, Canary, Rolling)
7. Monitoring et observabilité en production
8. Les erreurs classiques (et comment les éviter)

---

## 1. Docker : La Base de Tout

### Pourquoi Docker pour le ML ?

Si tu n'as qu'une chose à retenir : **Docker élimine le "ça marche sur ma machine"**. En ML, c'est encore plus critique qu'en développement classique parce que les dépendances sont un cauchemar :
- PyTorch 2.x avec CUDA 12.1 mais pas 12.4
- `numpy` qui casse silencieusement entre deux versions mineures
- Des librairies C++ compilées qui refusent de fonctionner sur une autre distribution

### Anatomie d'un Dockerfile ML bien fait

```dockerfile
# === Stage 1 : Builder ===
FROM python:3.11-slim AS builder

WORKDIR /build

# Copier UNIQUEMENT les fichiers de dépendances d'abord (cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# === Stage 2 : Runtime ===
FROM python:3.11-slim AS runtime

# Créer un utilisateur non-root (sécurité)
RUN useradd --create-home appuser
WORKDIR /home/appuser/app

# Copier les dépendances installées depuis le builder
COPY --from=builder /install /usr/local

# Copier le code applicatif
COPY --chown=appuser:appuser . .

# Healthcheck intégré
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser
EXPOSE 8000

CMD ["gunicorn", "app:create_app()", "-b", "0.0.0.0:8000", "-w", "4", "-k", "gthread"]
```

**Points clés :**
- **Multi-stage build** : le builder fait 1.2 Go, le runtime fait 400 Mo. En ML avec PyTorch, la différence peut être de 8 Go vs 2 Go.
- **Ordre des COPY** : les dépendances d'abord, le code ensuite. Comme les dépendances changent rarement, Docker utilise le cache et rebuild en 5 secondes au lieu de 5 minutes.
- **Utilisateur non-root** : ne fais JAMAIS tourner un conteneur en root en production. C'est la base.
- **Healthcheck** : Kubernetes en a besoin pour savoir si ton pod est vivant.

### Le .dockerignore (ne l'oublie pas)

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

Sans `.dockerignore`, ton build context inclut tes modèles de 4 Go et ton dossier `.git` de 500 Mo. Le build passe de 30 secondes à 10 minutes.

---

## 2. Docker Compose : L'Environnement Local et Staging

Docker Compose, c'est là où tu assembles les pièces. Un fichier, et tout le monde dans l'équipe a exactement le même environnement.

### docker-compose.yml pour un projet ML typique

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
     - ./app:/home/appuser/app/app  # Hot-reload en dev
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

**Anatomie du fichier :**

- **`app`** : le service principal. Le volume `./app:/home/appuser/app/app` monte le code source local dans le conteneur pour le hot-reload en dev (on modifie le code, l'app redémarre automatiquement). `depends_on` avec `condition: service_healthy` garantit que la base de données est prête avant de lancer l'app - sans ça, l'app crash au démarrage en essayant de se connecter à une DB qui n'existe pas encore. `deploy.resources.limits` empêche un modèle ML gourmand de consommer toute la RAM de la machine de dev.
- **`db`** : PostgreSQL avec un healthcheck intégré. Le fichier `init.sql` monté dans `docker-entrypoint-initdb.d/` est exécuté automatiquement au premier démarrage pour créer les tables. Le volume nommé `pgdata` persiste les données entre les redémarrages.
- **`redis`** : cache en mémoire avec une politique d'éviction LRU (Least Recently Used). Quand les 256 Mo sont pleins, Redis supprime les clés les moins utilisées. Utile pour cacher les embeddings ou les résultats d'inférence fréquents.
- **`nginx`** : reverse proxy devant l'app. En production, c'est lui qui termine le TLS, fait le load balancing, et sert les fichiers statiques. Le `:ro` (read-only) sur les volumes est une bonne pratique de sécurité.
- **`volumes`** en bas : les volumes nommés (`pgdata`, `model-cache`) persistent les données indépendamment du cycle de vie des conteneurs. Un `docker compose down` ne les supprime pas (il faut `docker compose down -v` pour ça).

### Les profils Docker Compose (la feature sous-cotée)

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
# Dev normal
docker compose up

# Dev + monitoring
docker compose --profile monitoring up

# Lancer les tests
docker compose --profile test run test-runner
```

---

## 3. GitHub Actions : Le Pipeline CI/CD

C'est ici que la magie opère. Chaque push déclenche une chaîne automatisée : tests, build, scan de sécurité, déploiement.

### Pipeline CI complet

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
  # === Job 1 : Tests ===
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

  # === Job 2 : Build & Push Docker Image ===
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

  # === Job 3 : Security Scan ===
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

**Explication du pipeline :**

Le workflow se déclenche sur chaque push vers `main` ou `develop` et sur les pull requests vers `main`. Il enchaîne trois jobs séquentiels :

- **Job 1 - Tests** : lance un conteneur PostgreSQL comme service auxiliaire (pour les tests d'intégration), installe les dépendances Python, exécute le linting avec `ruff` (formatage + style), puis les tests avec `pytest` et génère un rapport de couverture envoyé à Codecov. Si un test échoue, le pipeline s'arrête et le build n'a pas lieu.
- **Job 2 - Build & Push** : ne s'exécute que si les tests passent (`needs: test`). Configure Docker Buildx pour les builds multi-plateforme, s'authentifie au GitHub Container Registry (GHCR) avec le token automatique du workflow, puis build et push l'image Docker. Le `metadata-action` génère automatiquement les tags (SHA du commit, nom de branche, version sémantique). Le cache GitHub Actions (`cache-from/to: type=gha`) accélère les builds suivants en réutilisant les couches Docker inchangées.
- **Job 3 - Security** : scanne l'image Docker avec Trivy (scanner de vulnérabilités d'Aqua Security) pour détecter les CVE critiques et hautes dans les dépendances. Le résultat est exporté au format SARIF, intégrable directement dans l'onglet Security de GitHub.

### Pipeline CD (déploiement)

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
          # Deploy canary (10% du trafic)
          kubectl apply -f k8s/canary-deployment.yaml -n production
          sleep 60

          # Vérifier les métriques du canary
          ERROR_RATE=$(kubectl exec -n monitoring deploy/prometheus -- \
            promtool query instant 'rate(http_requests_total{status=~"5.."}[5m])')

          if [ $(echo "$ERROR_RATE > 0.05" | bc) -eq 1 ]; then
            echo "Canary failed - rolling back"
            kubectl rollout undo deployment/app-canary -n production
            exit 1
          fi

          # Promouvoir le canary en production complète
          kubectl set image deployment/app \
            app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n production
          kubectl rollout status deployment/app -n production --timeout=300s
```

**Explication du déploiement :**

Le workflow CD se déclenche automatiquement quand le pipeline CI réussit sur `main` (`workflow_run`). Il enchaîne deux environnements :

- **Staging** : `kubectl set image` met à jour l'image du Deployment avec le SHA du commit (tag unique et immuable). `kubectl rollout status` bloque jusqu'à ce que tous les pods soient mis à jour ou timeout après 5 minutes. Si le rollout échoue, le job échoue et la production n'est pas touchée.
- **Production** : ne se lance que si le staging a réussi. Utilise une stratégie canary : d'abord déployer la nouvelle version sur un petit sous-ensemble, attendre 60 secondes, puis vérifier le taux d'erreur via Prometheus. Si le taux dépasse 5%, rollback automatique et arrêt. Sinon, déploiement complet. Le `environment: production` dans GitHub permet d'ajouter des règles d'approbation manuelle (un humain doit valider avant le déploiement).

### Les secrets et la sécurité

Ne mets **JAMAIS** de secrets dans ton code. Utilise :
- **GitHub Secrets** pour les tokens, mots de passe, clés API
- **GitHub Environments** pour séparer staging/production avec des règles d'approbation
- **Sealed Secrets** ou **External Secrets Operator** côté Kubernetes

---

## 4. Kubernetes : L'Orchestration en Production

### Pourquoi Kubernetes ?

Docker Compose c'est bien pour le dev et les petits déploiements. Mais quand tu as besoin de :
- **Auto-scaling** : monter à 10 replicas quand le trafic explose
- **Self-healing** : redémarrer automatiquement un pod qui crash
- **Zero-downtime deployments** : déployer sans couper le service
- **Multi-environnement** : staging, production, avec isolation réseau

Là, Kubernetes devient indispensable.

### Les fichiers de déploiement essentiels

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

**Décryptage des sections clés :**

- **`replicas: 3`** - trois copies de l'app tournent en parallèle. Si un pod crash, les deux autres continuent à servir le trafic.
- **`strategy.rollingUpdate`** avec `maxUnavailable: 0` - pendant un déploiement, Kubernetes lance les nouveaux pods *avant* de tuer les anciens. Zéro coupure pour les utilisateurs.
- **`resources.requests`** vs **`limits`** - `requests` est le minimum garanti par le scheduler (le pod obtient au moins 500m CPU et 1 Go RAM). `limits` est le plafond - si le pod dépasse 4 Go, il est tué (OOM-kill). Pour le ML, mettez les limites généreuses car le chargement des modèles crée des pics de mémoire.
- **`livenessProbe`** - Kubernetes ping `/health` toutes les 10s. Si ça échoue 3 fois, le pod est tué et redémarré. Le `initialDelaySeconds: 30` laisse le temps de charger les modèles avant les vérifications.
- **`readinessProbe`** - similaire, mais contrôle si le pod reçoit du trafic. Un pod vivant mais pas prêt (modèle en cours de chargement) ne reçoit pas de requêtes.
- **`volumeMounts` + PVC** - les fichiers de modèles vivent sur un `PersistentVolumeClaim`, donc ils survivent aux redémarrages et n'ont pas besoin d'être re-téléchargés.

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

**Ce que ça fait :**

- Le **Service** est un load balancer interne. Il route le trafic vers tous les pods ayant le label `app: ml-api`. `ClusterIP` signifie qu'il n'est accessible que depuis l'intérieur du cluster - le trafic externe passe par l'Ingress.
- L'**Ingress** est le point d'entrée externe. Il termine le TLS (via `cert-manager` qui provisionne automatiquement les certificats Let's Encrypt), applique le rate limiting (100 req/min via les annotations nginx), et route le trafic HTTPS vers le Service interne sur le port 80.
- La séparation est importante : on peut changer d'Ingress controller (nginx → Traefik → Envoy) sans toucher à l'app. L'abstraction Service fait que les pods peuvent scaler et le trafic est automatiquement réparti.

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

**Comment fonctionne l'auto-scaling :**

- `minReplicas: 2` - toujours au moins 2 pods en fonctionnement (base de redondance).
- Quand le CPU moyen dépasse 70% sur tous les pods, Kubernetes en ajoute (jusqu'à 10). Quand ça baisse, il en retire.
- **`stabilizationWindowSeconds`** empêche le flapping : le scale-up attend 60s de charge soutenue avant d'ajouter des pods, le scale-down attend 300s (5 min) avant d'en retirer. Ça évite la boucle « scale up, pic terminé, scale down, nouveau pic, scale up ».
- Les `policies` limitent la vitesse de changement : max 2 pods ajoutés par minute, max 1 retiré par 2 minutes. Scale-up agressif, scale-down conservateur - c'est le défaut sûr pour les workloads ML où les cold starts sont coûteux.

### ConfigMaps et Secrets

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
# k8s/secret.yaml (en prod, utiliser Sealed Secrets)
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
stringData:
  database-url: "postgresql://user:pass@db:5432/appdb"
  api-key: "sk-..."
```

**La différence entre ConfigMap et Secret :**

- **ConfigMap** : configuration non-sensible, stockée en clair dans etcd. Idéal pour les paramètres applicatifs (niveau de log, timeouts, limites). Modifiable sans rebuild de l'image - un `kubectl rollout restart` suffit pour que les pods rechargent la config.
- **Secret** : données sensibles (mots de passe, clés API, certificats). Encodées en base64 dans etcd (pas chiffré par défaut - il faut activer le chiffrement at-rest sur le cluster). En production, utiliser **Sealed Secrets** (chiffrement asymétrique, les secrets chiffrés peuvent vivre dans Git) ou **External Secrets Operator** (synchronise depuis Vault, AWS Secrets Manager, etc.).
- Les deux s'injectent de la même façon dans les pods : via `envFrom` (toutes les clés deviennent des variables d'environnement) ou via `valueFrom.secretKeyRef` (clé par clé, comme dans le Deployment plus haut).

---

## 5. Helm & GitOps : Le Niveau Supérieur

### Helm : Templater Kubernetes

Quand tu as 15 fichiers YAML quasi identiques pour staging et production, Helm te sauve la vie.

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

**Comment ça marche :**

Un seul template Kubernetes (avec des variables `{{ .Values.replicaCount }}`, `{{ .Values.image.tag }}`, etc.) et des fichiers de valeurs par environnement. Le staging tourne avec 2 replicas, une image de la branche develop, et 2 Go de RAM. La production tourne avec 5 replicas, une version taggée stable (`v1.2.3`), et 4 Go de RAM. Un seul `helm upgrade --values values-production.yaml` applique toutes les différences d'un coup, sans dupliquer les fichiers YAML.

### GitOps avec ArgoCD

Le principe : **Git est la source de vérité**. Tu ne fais jamais `kubectl apply` manuellement. Tu push dans Git, ArgoCD détecte le changement et synchronise le cluster.

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

**Avantages :**
- Audit trail complet (chaque déploiement = un commit)
- Rollback = `git revert`
- Reproductibilité totale
- Drift detection (ArgoCD corrige les modifications manuelles)

---

## 6. Stratégies de Déploiement

### Rolling Update (par défaut dans K8s)

Remplace progressivement les anciens pods par les nouveaux. Simple, fiable, suffisant dans 80% des cas.

```
[v1] [v1] [v1]  -->  [v2] [v1] [v1]  -->  [v2] [v2] [v1]  -->  [v2] [v2] [v2]
```

### Blue-Green

Deux environnements identiques. Le trafic bascule d'un coup de l'ancien (blue) au nouveau (green). Rollback instantané : rebascule sur blue.

```
Blue  [v1] [v1] [v1]  <-- trafic
Green [v2] [v2] [v2]      (prêt)

# Switch
Blue  [v1] [v1] [v1]      (standby)
Green [v2] [v2] [v2]  <-- trafic
```

### Canary

Déploie la nouvelle version sur un petit pourcentage du trafic (5-10%), surveille les métriques, puis augmente progressivement.

```
[v1] [v1] [v1] [v1] [v1] [v1] [v1] [v1] [v1] [v2]  <-- 10% canary
                            ...surveillance...
[v2] [v2] [v2] [v2] [v2] [v2] [v2] [v2] [v2] [v2]  <-- 100% rollout
```

**Mon conseil** : commence par Rolling Update. Passe au Canary quand tu as du monitoring en place et que tu sais quelles métriques surveiller. Blue-Green si tu as besoin de rollback instantané et que tu peux te permettre de doubler les ressources.

---

## 7. Monitoring et Observabilité

Un déploiement sans monitoring, c'est comme conduire de nuit sans phares. Tu avances, mais tu ne vois pas le mur.

### La stack standard

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

**Les trois piliers de l'observabilité :**

- **Prometheus** : collecte les métriques. Il scrape (interroge) les endpoints `/metrics` de chaque service à intervalle régulier. Les métriques sont stockées en local sous forme de séries temporelles. Le fichier `prometheus.yml` définit quels services scraper et à quelle fréquence.
- **Grafana** : visualisation. Se connecte à Prometheus comme source de données et affiche les métriques sous forme de dashboards. Les dashboards pré-configurés dans `./monitoring/dashboards` sont chargés automatiquement au démarrage.
- **Loki** : agrégation de logs. C'est le "Prometheus des logs" - même interface de requête (LogQL), même intégration Grafana. Les applications envoient leurs logs à Loki via Promtail ou directement, et on peut les corréler avec les métriques dans Grafana.

### Les métriques essentielles pour du ML en prod

| Métrique | Seuil d'alerte | Pourquoi |
|:---|:---|:---|
| Latence p99 | > 2s | UX dégradée, timeout clients |
| Taux d'erreur 5xx | > 1% | Problème applicatif |
| CPU utilization | > 80% sustained | Besoin de scale-up |
| Memory utilization | > 85% | Risque OOM kill |
| Inference time | > 5s (ML spécifique) | Modèle trop lent, optimiser |
| Queue depth | > 100 requests | Backpressure, scale out |
| Pod restart count | > 3/heure | Crash loop, investiguer |

### Alerting avec Prometheus

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

**Comment lire ces règles :**

- **`HighErrorRate`** : `rate(http_requests_total{status=~"5.."}[5m])` calcule le taux de requêtes avec un code HTTP 5xx sur les 5 dernières minutes. Si ce taux dépasse 5% (`> 0.05`) pendant 2 minutes consécutives (`for: 2m`), l'alerte se déclenche en `critical`. Le `for` évite les faux positifs sur un pic isolé.
- **`HighLatency`** : `histogram_quantile(0.99, ...)` calcule le 99e percentile de la latence - c'est-à-dire que 99% des requêtes sont plus rapides que cette valeur. Si le p99 dépasse 2 secondes pendant 5 minutes, alerte `warning`. On utilise p99 plutôt que la moyenne car la moyenne masque les pics : une moyenne de 200ms peut cacher 1% de requêtes à 10s.

Ces alertes sont envoyées via Alertmanager vers Slack, PagerDuty, ou email selon la `severity`.

---

## 8. Les Erreurs Classiques (et Comment les Éviter)

Après quelques années dans le game, voici les erreurs que je vois le plus souvent :

### 1. Pas de healthcheck
Kubernetes ne sait pas si ton app est vivante. Il continue de router du trafic vers un pod mort. **Toujours** implémenter `/health` et `/ready`.

### 2. Images Docker de 8 Go
Utilise multi-stage builds. Sépare les dépendances de build des dépendances de runtime. Utilise les images `-slim` ou `-alpine`.

### 3. Secrets en dur dans le code
On le voit encore en 2025. Utilise des variables d'environnement, GitHub Secrets, Vault, ou Sealed Secrets. Point.

### 4. Pas de resource limits
Sans limites, un seul pod peut consommer toute la mémoire du nœud et faire crasher les autres. Mets TOUJOURS des `requests` et `limits`.

### 5. Déployer le vendredi
Je plaisante. Ou pas. Si ton pipeline CI/CD est solide et que tu as du canary en place, tu peux déployer n'importe quand. C'est justement le but de tout ce qu'on a mis en place.

### 6. Ignorer les logs
Centralise tes logs avec Loki, ELK, ou CloudWatch. Les `print()` dans la console d'un pod qui va se faire recycler dans 2 minutes, ça ne compte pas comme du logging.

### 7. Pas de rollback plan
Si tu ne peux pas rollback en 30 secondes, tu n'es pas prêt pour la prod. Kubernetes rollback, Git revert + ArgoCD sync, ou Blue-Green switch.

---

## Conclusion

Le DevOps pour le ML, c'est pas sexy. C'est du YAML, des pipelines, des healthchecks, et beaucoup de debugging de permissions Docker à 23h. Mais c'est ce qui fait la différence entre un side-project et un produit.

Le parcours type :

1. **Semaine 1** : Dockerfile propre + Docker Compose pour le dev local
2. **Semaine 2** : GitHub Actions CI (tests + build + push image)
3. **Semaine 3** : Déploiement Kubernetes (Deployment + Service + Ingress)
4. **Semaine 4** : Monitoring (Prometheus + Grafana) + alertes
5. **Mois 2** : Helm + ArgoCD + Canary deployments

Tu n'as pas besoin de tout faire d'un coup. Mais chaque étape te rapproche d'un système qui tourne tout seul - même à 3h du matin, un dimanche.

---

## Références

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

*Michail Berjaoui - Juin 2025*
