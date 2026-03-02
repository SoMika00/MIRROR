---
title: "Guide Pratique & Stratégique : Dimensionnement, Déploiement et Quantization de LLM"
date: 2025-02-20
tags: Quantization, VRAM, GPU, Inférence, Production, LLM, Deployment
summary: Standards architecturaux et meilleures pratiques pour le déploiement de LLM en production. Optimisation de l'inférence via quantization, dimensionnement VRAM sur GPU (H100, A100, RTX), choix des moteurs d'inférence (TRT-LLM, vLLM, llama.cpp), et arbre de décision pour le déploiement.
---

# Guide Pratique & Stratégique (2025) : Dimensionnement, Déploiement et Quantization de LLM

**Statut :** Validé | **Cible :** Ingénieurs MLOps, AI Architects, DevOps

Ce document documente les standards architecturaux et les meilleures pratiques pour le déploiement de Large Language Models (LLMs) en environnement de production. Il se concentre sur l'optimisation de l'inférence via la quantization, le dimensionnement de la VRAM sur GPU (NVIDIA H100, A100, RTX), et le choix des moteurs d'inférence (TRT-LLM, vLLM, llama.cpp).

*Pour MIRROR, j'ai suivi le chemin "CPU Only / Edge" : llama.cpp avec GGUF Q6_K sur un serveur Hetzner 64 Go. L'arbre de décision ci-dessous est le cadre que j'ai utilisé pour arriver à ce choix.*

## 1. Fondations : Gestion de la Mémoire d'un LLM

Pour maitriser la quantization, il est impératif de comprendre la répartition de l'empreinte mémoire d'un modèle lors de l'inférence. L'occupation de la VRAM se divise en trois piliers :

- **Poids (Weights)** : les paramètres du modèle appris durant l'entrainement. Ils dictent la taille minimale sur disque et en mémoire.
- **Activations** : les résultats intermédiaires calculés à chaque étape de l'inférence. Leur empreinte est généralement minime.
- **KV-Cache (Key-Value Cache)** : tenseurs de clés et valeurs du mécanisme d'attention, conservés au fil de la génération pour éviter de recalculer l'historique. Il croit de manière linéaire avec la longueur du contexte et le nombre de requêtes simultanées.

### La Formule du KV-Cache (avec GQA)

L'estimation de la mémoire allouée au KV-cache se calcule selon :

```
KV_bytes ≈ batch_size × sequence_length × num_layers × (2 × hidden_size) × (bytes_per_dtype / gqa_factor)
```

Exemple concret : pour Llama 3 70B en FP16, avec un batch de 4 et 8192 tokens de contexte, le KV-cache consomme environ 20 Go de VRAM en plus des poids du modèle.

### L'optimisation PagedAttention

Indépendamment du format numérique, **PagedAttention** (introduit par vLLM) optimise l'usage du KV-cache en le découpant en pages non contiguës. Cela réduit le gaspillage mémoire lié à la fragmentation de 60-80% à moins de 4%, permettant de maximiser l'in-flight batching.

## 2. Dimensionnement VRAM : Combien gagne-t-on concrètement ?

La règle d'or du dimensionnement repose sur la conversion du nombre de paramètres en octets :

- **FP16 (16 bits)** : 1 paramètre = 2 octets (multiplicateur x2)
- **FP8 / INT8 (8 bits)** : 1 paramètre = 1 octet (multiplicateur x1)
- **INT4 (4 bits)** : 1 paramètre = 0.5 octet (multiplicateur x0.5)

*Attention : toujours ajouter une marge de 15% à 20% pour le KV-cache et les activations.*

### Etude de cas sur GPU standards

| Modèle cible | Précision | VRAM Poids | VRAM Totale (avec contexte) | GPU minimum recommandé | Gain vs FP16 |
|:---|:---|:---|:---|:---|:---|
| **8B** (Llama 3 8B) | FP16 (Base) | ~16 Go | ~18-20 Go | 1x RTX 3090/4090 (24 Go) | Ref. |
| **8B** | FP8 / INT8 | ~8 Go | ~10-12 Go | 1x RTX 4060 Ti (16 Go) ou T4 | **-50%** |
| **8B** | INT4 (AWQ/GGUF) | ~4 Go | ~6-8 Go | 1x RTX 3060 (12 Go) | **-75%** |
| **70B** (Llama 3 70B) | FP16 (Base) | ~140 Go | ~160 Go | 2x H100 (80 Go) | Ref. |
| **70B** | FP8 / INT8 | ~70 Go | ~78 Go | **1x H100 (80 Go)** | **-50%** |
| **70B** | INT4 (AWQ/GGUF) | ~35 Go | ~42 Go | 1x A6000 (48 Go) | **-75%** |

### Le compromis : VRAM vs Vitesse vs Qualité

- **Gagner 50% de VRAM (FP8/INT8)** : la perte de qualité (perplexité) est quasi indétectable (< 1%). Sur un H100, le FP8 double la vitesse d'inférence grâce aux Tensor Cores spécialisés. C'est un gain absolu.
- **Gagner 75% de VRAM (INT4)** : la dégradation qualitative devient mesurable (nuances de style perdues, légères hallucinations sur du code/maths complexes). Le gain de vitesse est modéré car le calcul 4-bit n'est pas toujours accéléré nativement.

## 3. Les Méthodes de Quantization Clés

Plusieurs approches Post-Training Quantization (PTQ) dominent l'écosystème :

- **SmoothQuant (INT8 W8A8)** : lisse les valeurs extrêmes (outliers) des activations en transférant une partie de leur amplitude vers les poids via un rescaling. Permet une inférence 8-bit stable sans ré-entrainement, idéale sur architectures Ampere/Hopper. (Xiao et al., ICML 2023)

- **AWQ (Activation-aware Weight Quantization, INT4)** : identifie et protège le ~1% des poids les plus critiques (stockés en FP16), et quantifie les 99% restants en 4 bits. Offre le meilleur ratio qualité/compression pour des environnements contraints en VRAM. (Lin et al., MLSys 2024)

- **GPTQ (INT3/INT4)** : méthode one-shot utilisant l'approximation Hessienne pour compenser l'erreur de quantization bloc par bloc. Très populaire dans la sphère open-source. (Frantar et al., ICLR 2023)

- **GGUF (llama.cpp)** : format de quantization granulaire avec de nombreuses variantes (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0). Chaque variante offre un compromis différent entre taille et qualité. Q4_K_M est le sweet spot pour la plupart des usages CPU.

## 4. Comparatif des Moteurs d'Inférence (Serving)

### A. TensorRT-LLM (NVIDIA)
Compilateur et runtime ultra-optimisé spécifique aux GPU NVIDIA.
- **Avantages** : support FP8 natif sur H100, W8A8, AWQ, in-flight batching. Délivre le débit maximal et la latence minimale absolus (jusqu'a 4.6x plus de throughput sur H100 vs A100).
- **Inconvénients** : verrouillage matériel (CUDA uniquement), nécessite une compilation (build engine) préalable.

### B. vLLM (Open-Source)
Serveur Python/C++ développé par UC Berkeley, standard de l'industrie.
- **Avantages** : PagedAttention native, support dynamique du FP8/INT8, chargement à la volée. API Python compatible OpenAI.
- **Inconvénients** : latence pure sur une seule requête très légèrement supérieure à TRT-LLM, bien que le débit sous forte charge soit exceptionnel.

### C. llama.cpp / GGUF (CPU & Edge)
Implémentation minimaliste en C/C++ pour l'exécution locale.
- **Avantages** : agnostique au matériel (CPU, Apple Silicon, petits GPU). Supporte des formats de quantization très granulaires (Q4_K_M, Q8_0). SIMD acceleration (AVX2/AVX-512).
- **Inconvénients** : ne tire pas pleinement parti des Tensor Cores industriels. Déconseillé pour exploiter des serveurs data center à forte charge.

## 5. Arbre de Décision : Quelle Stratégie Déployer ?

1. **H100 End-to-End (Performance & Qualité Maximales)**
  - Méthode : FP8 (W8A8) + KV-Cache FP8
  - Outil : TensorRT-LLM ou vLLM (`--quantization fp8`)
  - Bénéfice : permet de faire tenir un 70B sur un seul H100 avec une vitesse et une qualité > 99% préservée

2. **Le Choix Universel (A100 / GPU standards)**
  - Méthode : INT8 SmoothQuant (W8A8) + KV-Cache FP8
  - Outil : vLLM (`--quantization int8`)
  - Bénéfice : idéal si le FP8 natif n'est pas supporté

3. **Haute Densité / Budget Réduit**
  - Méthode : INT4 AWQ (Poids uniquement) + KV-Cache FP8
  - Outil : vLLM (`--quantization awq`)
  - Bénéfice : divise les coûts de VRAM par 4. Parfait pour les chatbots internes

4. **CPU Only / Edge**
  - Méthode : GGUF Q4_K_M ou Q3_K_M
  - Outil : llama.cpp / llama-cpp-python
  - Bénéfice : aucun GPU nécessaire, fonctionne sur laptop ou serveur CPU (utilisé dans MIRROR)

## 6. Monitoring et SLIs en Production

| Métrique Cible | Seuil d'Alerte (Exemple) | Action Corrective |
|:---|:---|:---|
| Latence (Time To First Token) | p99 > 500ms | Scale-out ou bascule vers profil INT4 |
| Gaspillage KV-Cache | > 8% | Vérifier configuration PagedAttention |
| Erreurs OOM | > 0.5% des requêtes | Réduire le contexte max ou KV en FP8 |
| Throughput | < SLA (tokens/s) | Ajouter des replicas ou passer en FP8 |

## 7. Rappel sur la Conformité (Licensing)

La quantization est une transformation technique, elle **ne modifie en rien la licence source d'un modèle**. Un modèle "mergé" hérite systématiquement des restrictions de toutes ses composantes d'origine. Si l'usage commercial ou la redistribution sont interdits sur le modèle de base, ils le restent sur le GGUF ou le moteur TensorRT.

## 8. Cheat-Sheet des Commandes

```bash
# vLLM FP8 (H100)
vllm serve mistralai/Mistral-7B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384

# vLLM INT8 (A100 / GPU standards)
vllm serve meta-llama/Llama-3-70B-Instruct \
  --quantization int8 \
  --tensor-parallel-size 2

# vLLM AWQ INT4 (Budget)
vllm serve TheBloke/Llama-3-70B-Instruct-AWQ \
  --quantization awq \
  --max-model-len 8192

# TensorRT-LLM Build & Serve
trtllm-build \
  --checkpoint_dir /path/to/ckpt \
  --output_dir /path/to/engine_fp8 \
  --use_fp8 --use_fp8_kv_cache \
  --max_batch_size 16 --max_input_len 8192

# llama.cpp (CPU)
./llama-server -m model-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 0 -c 4096 -t $(nproc)
```

## Références

- **SmoothQuant** : Xiao et al. (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs*. ICML
- **AWQ** : Lin et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression*. MLSys
- **GPTQ** : Frantar et al. (2023). *GPTQ: Accurate Post-Training Quantization for GPT*. ICLR
- **vLLM & PagedAttention** : Kwon et al. (2023). *Efficient Memory Management for LLM Serving with PagedAttention*. SOSP
- **LLM.int8()** : Dettmers et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS

*Article complet disponible sur [GitHub](https://github.com/SoMika00/Quant_llm)*

*Michail Berjaoui - Février 2025*
