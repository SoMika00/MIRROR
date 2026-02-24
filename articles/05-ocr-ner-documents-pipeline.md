---
title: "Pipeline OCR + NER : Extraire des Données Structurées de Documents Scannés"
date: 2025-01-15
tags: OCR, NER, Documents, Pipeline, Production, VLM, Small LLM
summary: "Pipeline d'extraction de données à partir de documents scannés : OCR (Tesseract, PaddleOCR, DocTR), NER Transformer, optimisation avec petits LLMs pour l'extraction structurée, et déploiement en production."
---

# Pipeline OCR + NER : Extraire des Données Structurées de Documents Scannés

## Introduction

L'extraction automatique d'informations à partir de documents scannés (factures, contrats, formulaires) est un cas d'usage majeur en entreprise. La combinaison OCR (reconnaissance de texte) + NER (extraction d'entités nommées) permet de transformer des PDFs non structurés en données exploitables.

Cet article détaille la construction d'un pipeline robuste, de l'image brute à la donnée structurée en base de données.

## 1. Phase OCR : Du Pixel au Texte

### 1.1 Choix du Moteur OCR

Trois options principales selon le contexte :

- **Tesseract 5** : Open source, mature, bon pour les documents bien scannés avec mise en page simple. Supporte 100+ langues. Gratuit mais moins performant sur les layouts complexes.
- **PaddleOCR** : Open source (Baidu), excellent sur les documents multilingues et les layouts complexes (tableaux, colonnes). Détection de texte + reconnaissance en un seul pipeline.
- **Google Document AI / Azure Form Recognizer** : Solutions cloud, meilleures performances brutes, mais coût par page et dépendance cloud.

### 1.2 Préprocessing de l'Image

La qualité de l'OCR dépend directement de la qualité de l'image :

- **Deskew** : Corriger l'inclinaison du document (rotation automatique)
- **Binarisation** : Convertir en noir et blanc pour améliorer le contraste
- **Denoising** : Supprimer le bruit (taches, artefacts de scan)
- **Résolution** : Upscaler si la résolution est inférieure à 300 DPI

### 1.3 Post-processing OCR

Le texte brut de l'OCR contient des erreurs qu'il faut corriger :

- Correction orthographique contextuelle (dictionnaire métier)
- Reconstruction des tableaux (détection de lignes/colonnes)
- Préservation de l'ordre de lecture (multi-colonnes)

## 2. Phase NER : Du Texte aux Entités

### 2.1 Approches

- **NER classique (CRF/BiLSTM)** : Rapide, peu de données nécessaires, mais limitée aux patterns simples.
- **NER Transformer (BERT/CamemBERT fine-tuné)** : Meilleure compréhension du contexte, gère les entités ambiguës. Nécessite 500+ exemples annotés.
- **LLM en zero-shot/few-shot** : Pas de données d'entraînement nécessaires, mais plus lent et moins précis sur les entités très spécifiques.

### 2.2 Entités Typiques par Domaine

**Factures** : numéro de facture, date, montant HT/TTC, TVA, IBAN, nom du fournisseur, adresse.

**Contrats** : parties contractantes, dates de début/fin, montant, clauses spéciales, signatures.

**Formulaires médicaux** : nom du patient, date de naissance, numéro de sécurité sociale, diagnostic, traitements.

### 2.3 Annotation des Données

L'annotation est le goulot d'étranglement. Outils recommandés :

- **Label Studio** : Open source, interface web, support multi-annotateurs
- **Prodigy** (Explosion AI) : Annotation active learning, très efficace
- **Doccano** : Simple, adapté aux projets NER/classification

## 3. Architecture du Pipeline

```
Document (PDF/Image)
    │
    ▼
[Préprocessing Image]  →  Deskew, Binarisation, Denoising
    │
    ▼
[OCR Engine]           →  PaddleOCR / Tesseract
    │
    ▼
[Post-processing]      →  Correction, Reconstruction tableaux
    │
    ▼
[NER Model]            →  CamemBERT fine-tuné / LLM
    │
    ▼
[Validation]           →  Règles métier, score de confiance
    │
    ▼
[Sortie Structurée]    →  JSON / Base de données
```

## 4. Métriques et Évaluation

### OCR

- **CER (Character Error Rate)** : Pourcentage de caractères mal reconnus. Cible : <2% sur documents propres.
- **WER (Word Error Rate)** : Pourcentage de mots mal reconnus. Cible : <5%.

### NER

- **Precision** : Parmi les entités détectées, combien sont correctes ?
- **Recall** : Parmi les entités réelles, combien ont été détectées ?
- **F1-score** : Moyenne harmonique. Cible : >90% par type d'entité.

### Pipeline End-to-End

- **Extraction Accuracy** : Pourcentage de champs correctement extraits par document.
- **Throughput** : Documents traités par minute.

## 5. Optimisation avec des Petits LLMs

L'arrivée de petits LLMs performants (3-8B parametres) ouvre une alternative intéressante au pipeline OCR + NER classique.

### 5.1 Approche VLM (Vision Language Model)

Au lieu de OCR puis NER séparément, un VLM peut analyser directement l'image du document et extraire les informations en une seule passe :

- **MiniCPM-V 2.6** (8B) : comprend les layouts complexes, tableaux, et diagrammes. INT4 quantifié = ~4 Go RAM.
- **Qwen2-VL** (7B) : excellent sur les documents multilingues, supporte les résolutions arbitraires.
- **Florence-2** (0.7B) : ultra-léger, spécialisé OCR et détection d'objets.

Avantage : pas besoin de pipeline multi-étapes, le VLM gère OCR + compréhension + extraction en un seul appel.

### 5.2 LLM pour l'Extraction Structurée (JSON mode)

Apres l'OCR, au lieu d'un modèle NER dédié, utiliser un petit LLM en mode JSON :

```
System: Extrais les informations suivantes du texte OCR au format JSON:
{"numero_facture": "", "date": "", "montant_ttc": "", "fournisseur": ""}

User: [texte OCR brut de la facture]
```

Modèles adaptés :
- **Phi-4-mini (3.8B)** : rapide sur CPU, bon suivi d'instructions, format JSON fiable
- **Qwen 2.5 7B** : excellent multilingual, structured output
- **Llama 3.2 3B** : très rapide, suffisant pour l'extraction simple

### 5.3 Quand utiliser quelle approche ?

- **OCR + NER classique** : documents standardisés avec layout fixe, volume élevé (>1000/jour), latence critique (<1s)
- **OCR + petit LLM** : schemas d'extraction variables, besoin de flexibilité, volume modéré
- **VLM direct** : documents avec layouts complexes (tableaux, graphiques), quand la qualité prime sur la latence

## 6. Leçons Apprises en Production

1. **La qualité du scan est reine** : imposer des contraintes de résolution (300+ DPI) réduit 80% des erreurs OCR.
2. **Les tableaux restent difficiles** : les modèles layout-aware (LayoutLMv3, DocTR, Table Transformer) aident mais ne sont pas parfaits. Prévoir un fallback VLM pour les cas complexes.
3. **Le feedback loop est essentiel** : permettre aux utilisateurs de corriger les extractions et réinjecter ces corrections dans le dataset d'entrainement.
4. **Les cas limites sont la norme** : documents tournés, multi-pages, tampons sur le texte, écriture manuscrite. Prévoir des fallbacks humains avec scoring de confiance.
5. **Les petits LLMs changent la donne** : pour les volumes modérés, un Phi-4-mini en JSON mode remplace avantageusement un pipeline NER dédié avec moins de maintenance.

## Conclusion

Un pipeline OCR+NER de production est un système qui demande une attention constante à la qualité des données d'entrée, une évaluation rigoureuse à chaque étape, et un feedback loop avec les utilisateurs. La combinaison PaddleOCR + CamemBERT fine-tuné reste excellente pour le volume. Pour plus de flexibilité, les petits LLMs (Phi-4-mini, Qwen 2.5) en mode JSON offrent une extraction structurée sans entrainer de modèle NER dédié. Les VLMs (MiniCPM-V) représentent l'avenir pour les documents complexes.
