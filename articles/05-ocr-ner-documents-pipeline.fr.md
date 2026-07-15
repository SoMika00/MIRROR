---
title: "De Millions de PDF à des Données Structurées : Pipeline OCR + NER en Production"
date: 2025-01-15
tags: OCR, NER, Documents, Pipeline, Production, VLM, Small LLM, Dataset Creation
summary: "Comment transformer des millions de documents non structurés (PDF scannés, factures, contrats) en données structurées exploitables. OCR, NER, VLM, création de datasets, et les défis réels de la production."
---

# De Millions de PDF à des Données Structurées : Pipeline OCR + NER en Production

## Le problème

Chaque entreprise a le même problème : des millions de documents scannés (factures, contrats, formulaires, pièces d'identité) qui contiennent des informations critiques, mais piégées dans du pixel. Des équipes entières passent leurs journées à recopier des chiffres dans des tableurs. C'est lent, coûteux, et bourré d'erreurs humaines.

La promesse : prendre un PDF scanné, en extraire automatiquement les champs utiles (montant, date, nom, IBAN), et les injecter directement dans une base de données ou un ERP. À l'échelle, ce sont des milliers d'heures de saisie manuelle économisées.

La réalité : c'est un des problèmes les plus sous-estimés en data engineering. La qualité des scans est catastrophique, les layouts sont imprévisibles, et les cas limites représentent 30 % du volume.

## 1. La puissance (et les limites) de l'OCR

L'OCR (Optical Character Recognition) est la première brique : transformer une image en texte brut.

### Les moteurs

Plusieurs options selon le contexte :

- **Tesseract 5** : open source, mature, 100+ langues. Performant sur les documents propres avec layout simple. Gratuit, mais se noie sur les tableaux complexes et les scans de mauvaise qualité.
- **PaddleOCR** : open source (Baidu), le meilleur rapport qualité/prix. Excellent sur les layouts complexes (tableaux, colonnes, formulaires). Détection + reconnaissance en un seul pipeline. Mon choix par défaut pour la production.
- **DocTR** (Mindee) : open source, à base de Transformers. Très bon sur les documents européens. Plus lourd que PaddleOCR mais plus précis sur les cas difficiles.
- **Google Document AI / Azure Form Recognizer** : solutions cloud, meilleures performances brutes, mais coût par page et dépendance cloud. À considérer pour les volumes massifs avec budget.

### Le preprocessing : 80 % du travail

La qualité de l'OCR dépend directement de la qualité de l'image. Un bon preprocessing transforme un OCR médiocre en OCR excellent :

- **Deskew** : corriger l'inclinaison. Un document tourné de 2 degrés suffit à faire chuter la précision de 15 %.
- **Binarisation adaptative** : convertir en noir et blanc avec un seuil local. Les scans avec éclairage inégal en ont besoin.
- **Denoising** : supprimer taches, artefacts, tampons qui chevauchent le texte.
- **Résolution** : upscaler tout ce qui est en dessous de 300 DPI. En dessous, les petits caractères deviennent illisibles pour l'OCR.

### Les défis réels

Ce que les tutos ne disent pas :
- Les tampons et signatures par-dessus le texte détruisent la détection
- Les tableaux sans bordures visibles sont un cauchemar (le moteur mélange les colonnes)
- L'écriture manuscrite reste un problème ouvert pour l'OCR classique
- Les PDF « numériques » (non scannés) contiennent déjà du texte extractible : toujours vérifier avant de lancer l'OCR

### Post-processing

Le texte brut de l'OCR contient des erreurs qu'il faut corriger :
- Correction orthographique contextuelle avec dictionnaire métier
- Reconstruction des tableaux (détection de lignes/colonnes via des modèles layout-aware comme LayoutLMv3 ou Table Transformer)
- Préservation de l'ordre de lecture (multi-colonnes, headers/footers)

## 2. Le NER : du texte brut aux entités exploitables

Le NER (Named Entity Recognition) est la deuxième brique : identifier et extraire les entités utiles dans le texte OCR.

### Les approches

- **NER classique (CRF/BiLSTM)** : rapide, peu de données nécessaires, mais limité aux patterns simples et réguliers.
- **NER Transformer (BERT/CamemBERT fine-tuné)** : meilleure compréhension du contexte, gère les entités ambiguës. Nécessite 500+ exemples annotés, mais le gain en précision est massif. Mon choix pour la production en français.
- **LLM en zero-shot/few-shot** : pas de données d'entraînement nécessaires, flexible sur les schémas d'extraction. Plus lent et moins précis sur les entités très spécifiques, mais imbattable pour le prototypage.

### Entités typiques par domaine

**Factures** : numéro de facture, date d'émission, montant HT/TTC, TVA, IBAN, nom du fournisseur, adresse de facturation.

**Contrats** : parties contractantes, dates de début/fin, montant, clauses spéciales, conditions de résiliation.

**Pièces d'identité** : nom, prénom, date de naissance, numéro de document, date d'expiration, nationalité. C'est ce que j'ai automatisé chez SOMA pour la validation de documents.

**Formulaires médicaux** : nom du patient, numéro de sécurité sociale, diagnostic, traitements prescrits.

### L'annotation : le goulot d'étranglement

Pas de NER performant sans données annotées de qualité. Outils :
- **Label Studio** : open source, interface web, multi-annotateurs, le plus complet
- **Prodigy** (Explosion AI) : annotation avec active learning, très efficace pour réduire le volume nécessaire
- **Doccano** : simple, adapté aux projets NER/classification de petite taille

## 3. L'architecture du pipeline

```
Documents (PDF/Image, par milliers)
    |
    v
[Triage]               -->  PDF natif ? Extraire le texte directement.
    |                        PDF scanné ? Envoyer à l'OCR.
    v
[Preprocessing Image]  -->  Deskew, Binarisation, Denoising, Upscale
    |
    v
[OCR Engine]           -->  PaddleOCR / DocTR / Tesseract
    |
    v
[Post-processing]      -->  Correction, Reconstruction tableaux
    |
    v
[NER / LLM Extract]   -->  CamemBERT fine-tuné / LLM JSON mode
    |
    v
[Validation]           -->  Règles métier, score de confiance, seuils
    |
    v
[Sortie Structurée]    -->  JSON / PostgreSQL / API
    |
    v
[Feedback Loop]        -->  Corrections humaines réinjectées dans le training set
```

## 4. Métriques et évaluation

### OCR
- **CER (Character Error Rate)** : pourcentage de caractères mal reconnus. Cible : <2 % sur documents propres, <5 % sur scans médiocres.
- **WER (Word Error Rate)** : pourcentage de mots mal reconnus. Cible : <5 %.

### NER
- **Précision** : parmi les entités détectées, combien sont correctes ?
- **Recall** : parmi les entités réelles, combien ont été détectées ?
- **F1-score** : moyenne harmonique. Cible : >90 % par type d'entité en production.
- **Exact match vs partial match** : un montant « 1 234,56 » détecté comme « 1 234 » est un partial match. En production, seul l'exact match compte.

### Pipeline end-to-end
- **Extraction Accuracy** : pourcentage de champs correctement extraits par document. C'est LA métrique business.
- **Throughput** : documents traités par minute. Critique pour le dimensionnement.
- **Taux de rejet** : pourcentage de documents envoyés en review humaine (score de confiance trop bas).

## 5. Les petits LLM changent la donne

L'arrivée de petits LLM performants (3-8B paramètres) ouvre une alternative puissante au pipeline NER classique.

### Approche VLM (Vision Language Model)

Au lieu de faire OCR puis NER séparément, un VLM analyse directement l'image et extrait les informations en une seule passe :

- **MiniCPM-V 2.6** (8B) : comprend les layouts complexes, tableaux, diagrammes. Quantifié INT4 = ~4 Go RAM.
- **Qwen2-VL** (7B) : excellent sur les documents multilingues, résolutions arbitraires.
- **Florence-2** (0.7B) : ultra-léger, spécialisé OCR et détection d'objets.

L'avantage est énorme : pas de pipeline multi-étapes, le VLM gère OCR + compréhension + extraction en un seul appel. Moins de code, moins de bugs, moins de maintenance.

### LLM pour l'extraction structurée (JSON mode)

Après l'OCR, au lieu d'un modèle NER dédié, utiliser un petit LLM en mode JSON :

```
System: Extrais les informations suivantes du texte OCR au format JSON:
{"numero_facture": "", "date": "", "montant_ttc": "", "fournisseur": ""}

User: [texte OCR brut de la facture]
```

Modèles adaptés :
- **Phi-4-mini (3.8B)** : rapide sur CPU, bon suivi d'instructions, JSON fiable
- **Qwen 2.5 7B** : excellent en multilingue, structured output
- **Llama 3.2 3B** : très rapide, suffisant pour l'extraction simple

### Création de datasets : le LLM comme annotateur

C'est là que ça devient vraiment puissant. Au lieu d'annoter manuellement des milliers de documents :

1. Passer un échantillon (100-200 docs) dans un gros LLM (GPT-4, Claude) avec un prompt d'extraction détaillé
2. Vérifier manuellement les résultats, corriger les erreurs
3. Utiliser ce dataset comme base d'entraînement pour un modèle NER dédié ou un petit LLM fine-tuné
4. Le modèle fine-tuné traite les millions de documents restants à haute vitesse

C'est de la création de dataset pure et dure : le gros LLM « devine » les annotations, l'humain valide, et le petit modèle scale. On passe de semaines d'annotation manuelle à quelques jours.

### Quand utiliser quelle approche ?

- **OCR + NER classique** : documents standardisés avec layout fixe, volume élevé (>1000/jour), latence critique (<1 s)
- **OCR + petit LLM** : schémas d'extraction variables, besoin de flexibilité, volume modéré
- **VLM direct** : documents avec layouts complexes (tableaux, graphiques), quand la qualité prime sur la latence
- **LLM-as-annotator + NER fine-tuné** : quand vous avez des millions de documents et peu de données annotées

## 6. Production : ce qui fait la différence

### Scoring de confiance

Chaque extraction a un score de confiance. En dessous du seuil (typiquement 0,85), le document part en review humaine. C'est non négociable en production : mieux vaut ralentir que d'injecter des données fausses dans le système.

### Feedback loop

Les corrections humaines sont réinjectées dans le dataset d'entraînement. Le modèle s'améliore en continu. Après 3 mois de production, notre F1 est passé de 0,88 à 0,94 uniquement grâce au feedback loop.

### Scaling

Pour traiter des milliers de documents par heure :
- Paralléliser l'OCR (CPU-bound, scale linéaire avec les cœurs)
- Batcher les appels NER/LLM
- Queue (Redis/Kafka) pour découpler l'ingestion du traitement
- Monitoring du throughput et du taux de rejet en temps réel

### Les cas limites sont la norme

En production, 30 % des documents sont des cas limites : documents tournés, multi-pages avec layouts différents, tampons sur le texte, écriture manuscrite, qualité de scan dégradée. Le pipeline doit avoir des fallbacks clairs : VLM pour les cas complexes, review humaine pour les cas critiques.

## Conclusion

Transformer des millions de PDF non structurés en données structurées est un problème d'ingénierie, pas de recherche. Les briques sont là : PaddleOCR pour le texte, CamemBERT pour le NER, les petits LLM pour la flexibilité, les VLM pour les cas complexes.

Ce qui fait la différence entre un prototype et un système de production, c'est le preprocessing, le scoring de confiance, le feedback loop, et la capacité à gérer les 30 % de cas limites sans faire crasher le pipeline.

*Cet article s'appuie largement sur mon expérience chez SOMA, où j'ai automatisé la validation de documents d'identité avec des pipelines OCR + NER. Traiter des milliers de documents par jour m'a appris que les 30 % de cas limites ne sont pas des exceptions - c'est le vrai problème à résoudre.*

*Michail Berjaoui - Janvier 2025*
