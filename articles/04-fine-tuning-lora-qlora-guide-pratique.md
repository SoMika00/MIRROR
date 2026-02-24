---
title: "Fine-tuning LoRA/QLoRA : Guide Pratique pour Adapter un LLM à votre Domaine"
date: 2024-12-05
tags: Fine-tuning, LoRA, QLoRA, PEFT, Unsloth, Dataset
summary: "Guide étape par étape pour fine-tuner un LLM open source avec LoRA et QLoRA : préparation du dataset, configuration des hyperparamètres, entraînement, évaluation, et déploiement de l'adaptateur."
---

# Fine-tuning LoRA/QLoRA : Guide Pratique pour Adapter un LLM à votre Domaine

## Introduction

Le prompting a ses limites. Quand un LLM ne produit pas les résultats attendus malgré des prompts bien construits, le fine-tuning devient nécessaire. Grâce à LoRA (Low-Rank Adaptation) et QLoRA (Quantized LoRA), il est désormais possible d'adapter un modèle de 7B à 70B paramètres sur un seul GPU consumer (24 Go VRAM).

Cet article est un guide pratique issu de plusieurs fine-tunings réalisés en production, couvrant la préparation des données jusqu'au déploiement.

## 1. Quand Fine-tuner (et Quand Ne Pas le Faire)

### Le fine-tuning est justifié quand :

- Le modèle ne maîtrise pas votre **terminologie métier** (médical, juridique, financier)
- Vous avez besoin d'un **format de sortie strict** (JSON spécifique, rapport structuré)
- Le **few-shot prompting** ne donne pas de résultats consistants
- Vous voulez réduire la **latence** en éliminant les longs prompts système

### Le fine-tuning n'est PAS la solution quand :

- Vous n'avez pas **au moins 500 exemples de qualité**
- Le problème vient d'un mauvais **retrieval** dans votre pipeline RAG
- Un meilleur **prompt engineering** pourrait suffire
- Vous n'avez pas de **métriques d'évaluation** définies

## 2. Préparation du Dataset

La qualité du dataset est le facteur n°1 de succès. Garbage in, garbage out.

### Format des données

Le format standard est une liste de conversations en JSON :

```json
{
  "conversations": [
    {"role": "system", "content": "Tu es un assistant juridique spécialisé en droit du travail français."},
    {"role": "user", "content": "Quels sont les délais de préavis pour un CDI de 3 ans ?"},
    {"role": "assistant", "content": "Pour un salarié en CDI avec 3 ans d'ancienneté, le préavis légal est de 2 mois..."}
  ]
}
```

### Taille du dataset

- **Minimum viable** : 500 exemples pour un changement de style/format
- **Recommandé** : 1 000-5 000 exemples pour une adaptation de domaine
- **Optimal** : 5 000-10 000 exemples pour un changement de comportement profond

### Nettoyage critique

- Supprimer les doublons et quasi-doublons
- Vérifier la cohérence des réponses
- Équilibrer les catégories si classification
- Valider manuellement un échantillon de 10%

## 3. Configuration LoRA

### Hyperparamètres clés

- **Rank (r)** : Dimension des matrices d'adaptation. r=8 pour du style, r=32-64 pour du domaine. Plus le rank est élevé, plus le modèle peut apprendre, mais plus le risque d'overfitting augmente.
- **Alpha** : Facteur de scaling. Règle empirique : alpha = 2 × rank.
- **Target modules** : Quelles couches adapter. Par défaut les couches d'attention (q_proj, v_proj). Pour plus de capacité, ajouter k_proj, o_proj, gate_proj, up_proj, down_proj.
- **Dropout** : 0.05-0.1 pour éviter l'overfitting.

### Configuration QLoRA

QLoRA charge le modèle de base en 4-bit (NF4) et n'entraîne que les adaptateurs LoRA en FP16/BF16. Cela divise la VRAM par 4 :

- Llama 3.1 8B : ~6 Go VRAM (vs 16 Go en FP16)
- Llama 3.1 70B : ~40 Go VRAM (vs 140 Go en FP16)

## 4. Entraînement

### Avec Unsloth (recommandé, 2x plus rapide)

Unsloth optimise automatiquement les kernels CUDA pour LoRA/QLoRA. Il supporte Llama, Mistral, Phi, Qwen, Gemma.

### Paramètres d'entraînement

- **Learning rate** : 2e-4 à 5e-5 (plus bas pour les gros modèles)
- **Epochs** : 1-3 (rarement plus, risque de catastrophic forgetting)
- **Batch size** : Le plus grand possible dans la VRAM (gradient accumulation si nécessaire)
- **Warmup** : 5-10% des steps
- **Scheduler** : Cosine decay

### Signaux d'alerte pendant l'entraînement

- **Loss qui ne descend pas** : Learning rate trop bas, ou rank trop faible
- **Loss qui descend trop vite** : Overfitting, réduire les epochs
- **Loss qui oscille** : Learning rate trop élevé

## 5. Fine-tuning pour RAG

Un cas d'usage de plus en plus courant : fine-tuner un LLM pour qu'il exploite mieux le contexte récupéré par un pipeline RAG.

### Pourquoi fine-tuner pour le RAG ?

Meme les meilleurs LLMs out-of-the-box ont des limites en RAG :
- Ils peuvent **ignorer le contexte** fourni et répondre de mémoire (hallucination)
- Ils ne savent pas toujours **citer leurs sources** de manière fiable
- Le format de citation peut être **inconstant** (parfois en footnote, parfois inline)
- Ils peuvent **inventer des passages** qui ne sont pas dans le contexte

### Dataset pour RAG fine-tuning

Le dataset doit enseigner au modèle à :
1. **Toujours baser sa réponse sur le contexte fourni** (et le dire quand le contexte ne contient pas la réponse)
2. **Citer les sources** avec un format cohérent (`[Source: nom, p.X]`)
3. **Synthétiser** plusieurs passages plutôt que copier-coller

Format type :

```json
{
  "conversations": [
    {"role": "system", "content": "Réponds en te basant uniquement sur le contexte fourni. Cite tes sources."},
    {"role": "user", "content": "Contexte:\n[Source: rapport.pdf, p.12] Le chiffre d'affaires a augmenté de 15%...\n[Source: bilan.pdf, p.3] La marge nette est de 8.2%...\n\nQuestion: Quelle est la santé financière de l'entreprise ?"},
    {"role": "assistant", "content": "L'entreprise montre une bonne santé financière : le chiffre d'affaires a progressé de 15% [Source: rapport.pdf, p.12] avec une marge nette de 8.2% [Source: bilan.pdf, p.3]."}
  ]
}
```

### Génération de données synthétiques

Quand vous n'avez pas assez de paires question-réponse, utilisez un LLM plus puissant (GPT-4, Claude) pour en générer :

1. Prenez vos vrais documents chunked
2. Pour chaque chunk, demandez au LLM de générer 3-5 questions auxquelles le chunk répond
3. Demandez-lui de générer les réponses avec citations
4. Validez manuellement un échantillon de 10-20%

Cette approche (distillation) permet de passer de 100 exemples manuels à 2000+ exemples de qualité.

## 6. Évaluation

Ne jamais évaluer uniquement sur la loss d'entrainement. Métriques à suivre :

- **Perplexité** sur un jeu de test held-out (mesure la "surprise" du modèle)
- **Évaluation humaine** : échantillon de 50-100 réponses notées manuellement sur des critères précis (pertinence, format, factualité)
- **LLM-as-judge** : utiliser GPT-4 ou Claude pour scorer les réponses sur des rubrics définis. Plus scalable que l'évaluation humaine, bonne corrélation.
- **Métriques task-specific** : accuracy pour classification, BLEU/ROUGE pour génération, F1 pour NER
- **RAGAS** (pour RAG) : faithfulness, answer relevancy, context precision, citation correctness
- **Regression check** : vérifier que le modèle n'a pas perdu ses capacités générales (tester sur MMLU, HellaSwag)

### A/B testing en production

Le test ultime : déployer le modèle fine-tuné en parallèle du modèle de base et comparer les métriques utilisateur (thumbs up/down, reformulations, temps passé sur la réponse).

## 7. Alignment post-fine-tuning (DPO)

Apres le SFT (Supervised Fine-Tuning), vous pouvez aller plus loin avec **DPO (Direct Preference Optimization)** : fournir des paires (réponse préférée, réponse rejetée) pour que le modèle apprenne vos préférences de qualité. DPO est plus simple que RLHF (pas besoin d'un reward model séparé) et fonctionne bien avec TRL de HuggingFace.

## 8. Déploiement

### Merge de l'adaptateur

Fusionner l'adaptateur LoRA dans le modèle de base pour éliminer l'overhead d'inférence :

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")
```

### Quantization post-merge

Convertir le modèle mergé en GGUF pour le déployer avec llama.cpp :

```bash
python convert_hf_to_gguf.py ./merged-model --outtype f16
./llama-quantize merged-model-f16.gguf merged-model-Q4_K_M.gguf Q4_K_M
```

Le modèle GGUF résultant est directement utilisable dans llama.cpp, llama-cpp-python, ou Ollama.

### Adaptateurs empilables

Un avantage majeur de LoRA : on peut maintenir plusieurs adaptateurs pour le meme modèle de base. Un adaptateur "juridique", un "médical", un "RAG-optimisé". Le routage se fait au niveau de l'application, le modèle de base n'est chargé qu'une fois en mémoire.

## Conclusion

Le fine-tuning avec LoRA/QLoRA démocratise l'adaptation des LLM. Les clés du succes : un dataset de qualité (500+ exemples minimum), une évaluation rigoureuse (jamais juste la loss), et une approche itérative. Commencez avec un petit dataset et un rank faible, mesurez avec des métriques concrètes, puis augmentez progressivement. Pour le RAG, le fine-tuning améliore significativement la capacité du modèle à utiliser le contexte et à citer ses sources.
