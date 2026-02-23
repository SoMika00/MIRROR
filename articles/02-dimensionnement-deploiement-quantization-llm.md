---
title: "Guide Pratique & Stratégique : Dimensionnement, Déploiement et Quantization de LLM"
date: 2025-02-20
tags: LLM, Quantization, vLLM, TensorRT-LLM, VRAM, Déploiement
summary: Standards architecturaux et meilleures pratiques pour le déploiement de LLM en production — optimisation de l'inférence via quantization, dimensionnement VRAM sur GPU (H100, A100, RTX), et choix des moteurs d'inférence.
---

# Guide Pratique & Stratégique (2025) : Dimensionnement, Déploiement et Quantization de LLM

**Statut :** Validé | **Cible :** Ingénieurs MLOps, AI Architects, DevOps

Ce document documente les standards architecturaux et les meilleures pratiques pour le déploiement de Large Language Models (LLMs) en environnement de production. Il se concentre sur l'optimisation de l'inférence via la quantization, le dimensionnement de la VRAM sur GPU (NVIDIA H100, A100, RTX), et le choix des moteurs d'inférence (TRT-LLM, vLLM).

## 1. Fondations : Gestion de la Mémoire d'un LLM

L'occupation de la VRAM se divise en trois piliers :

- **Poids (Weights)** — Les paramètres appris durant l'entraînement, dictant la taille minimale
- **Activations** — Résultats intermédiaires calculés à chaque étape d'inférence (empreinte minime)
- **KV-Cache** — Tenseurs clés/valeurs du mécanisme d'attention, croissant linéairement avec le contexte et le nombre de requêtes simultanées

### La Formule du KV-Cache (avec GQA)

Le KV-Cache est souvent le facteur limitant en production. Avec l'optimisation PagedAttention (vLLM), la gestion mémoire est fragmentée en blocs de taille fixe, éliminant la fragmentation et permettant le partage entre séquences.

## 2. Dimensionnement VRAM

| Quantization | Taille (14B) | Perplexité Delta | Impact Vitesse |
|-------------|-------------|-----------------|---------------|
| FP16        | ~28 Go      | Baseline        | Baseline      |
| FP8         | ~14 Go      | +0.05           | +15%          |
| Q4_K_M      | ~9 Go       | +0.4            | +25%          |
| Q3_K_M      | ~7 Go       | +1.2            | +35%          |

## 3. Les Méthodes de Quantization Clés

- **SmoothQuant (INT8 W8A8)** — Xiao et al., ICML 2023
- **AWQ (INT4)** — Lin et al., MLSys 2024, activation-aware
- **GPTQ (INT3/INT4)** — Frantar et al., ICLR 2023
- **LLM.int8()** — Dettmers et al., NeurIPS 2022

## 4. Comparatif des Moteurs d'Inférence

### A. TensorRT-LLM (NVIDIA)
Optimisé pour GPU NVIDIA, support FP8 natif sur Hopper, batch scheduling avancé.

### B. vLLM (Open-Source)
PagedAttention pour gestion mémoire efficace, continuous batching, support multi-GPU.

### C. llama.cpp / GGUF (CPU & Edge)
Idéal pour déploiement sans GPU, quantification GGUF, SIMD acceleration (AVX2/AVX-512).

## 5. Arbre de Décision

1. **GPU Hopper (H100)** → TensorRT-LLM FP8
2. **GPU Ada/Ampere** → vLLM + AWQ INT4
3. **CPU Only** → llama.cpp + GGUF Q4_K_M
4. **Edge / Embarqué** → llama.cpp + Q3 ou modèle distillé 3B

## 6. Cheat-Sheet des Commandes

```bash
# vLLM FP8
vllm serve mistralai/Mistral-7B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384

# TensorRT-LLM Build & Serve
trtllm-build \
  --checkpoint_dir /path/to/ckpt \
  --output_dir /path/to/engine_fp8 \
  --use_fp8 --use_fp8_kv_cache \
  --max_batch_size 16 --max_input_len 8192
```

## Références

- **SmoothQuant** — Xiao et al. (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs*. ICML
- **AWQ** — Lin et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression*. MLSys
- **GPTQ** — Frantar et al. (2023). *GPTQ: Accurate Post-Training Quantization for GPT*. ICLR
- **vLLM & PagedAttention** — Kwon et al. (2023). *Efficient Memory Management for LLM Serving*. SOSP

*Article complet disponible sur [GitHub](https://github.com/SoMika00/Quant_llm)*
