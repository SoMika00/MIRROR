---
title: "De Millions de PDFs a des Donnees Structurees : Pipeline OCR + NER en Production"
date: 2025-01-15
tags: OCR, NER, Documents, Pipeline, Production, VLM, Small LLM, Dataset Creation
summary: "Comment transformer des millions de documents non structures (PDFs scannes, factures, contrats) en donnees structurees exploitables. OCR, NER, VLMs, creation de datasets, et les defis reels de la production."
---

# De Millions de PDFs a des Donnees Structurees : Pipeline OCR + NER en Production

## Le probleme

Chaque entreprise a le meme probleme : des millions de documents scannes (factures, contrats, formulaires, pieces d'identite) qui contiennent des informations critiques, mais piégées dans du pixel. Des equipes entieres passent leurs journees a recopier des chiffres dans des tableurs. C'est lent, couteux, et bourre d'erreurs humaines.

La promesse : prendre un PDF scanne, en extraire automatiquement les champs utiles (montant, date, nom, IBAN), et les injecter directement dans une base de donnees ou un ERP. A l'echelle, c'est des milliers d'heures de saisie manuelle economisees.

La realite : c'est un des problemes les plus sous-estimes en data engineering. La qualite des scans est catastrophique, les layouts sont imprevisibles, et les cas limites representent 30% du volume.

## 1. La puissance (et les limites) de l'OCR

L'OCR (Optical Character Recognition) est la premiere brique : transformer une image en texte brut.

### Les moteurs

Trois options selon le contexte :

- **Tesseract 5** : open source, mature, 100+ langues. Performant sur les documents propres avec layout simple. Gratuit, mais se noie sur les tableaux complexes et les scans de mauvaise qualite.
- **PaddleOCR** : open source (Baidu), le meilleur rapport qualite/prix. Excellent sur les layouts complexes (tableaux, colonnes, formulaires). Detection + reconnaissance en un seul pipeline. Mon choix par defaut pour la production.
- **DocTR** (Mindee) : open source, Transformer-based. Tres bon sur les documents europeens. Plus lourd que PaddleOCR mais plus precis sur les cas difficiles.
- **Google Document AI / Azure Form Recognizer** : solutions cloud, meilleures performances brutes, mais cout par page et dependance cloud. A considerer pour les volumes massifs avec budget.

### Le preprocessing : 80% du travail

La qualite de l'OCR depend directement de la qualite de l'image. Un bon preprocessing transforme un OCR mediocre en OCR excellent :

- **Deskew** : corriger l'inclinaison. Un document tourne de 2 degres suffit a faire chuter la precision de 15%.
- **Binarisation adaptative** : convertir en noir et blanc avec un seuil local. Les scans avec eclairage inegal en ont besoin.
- **Denoising** : supprimer taches, artefacts, tampons qui chevauchent le texte.
- **Resolution** : upscaler tout ce qui est en dessous de 300 DPI. En dessous, les petits caracteres deviennent illisibles pour l'OCR.

### Les defis reels

Ce que les tutos ne disent pas :
- Les tampons et signatures par-dessus le texte detruisent la detection
- Les tableaux sans bordures visibles sont un cauchemar (le moteur melange les colonnes)
- L'ecriture manuscrite reste un probleme ouvert pour l'OCR classique
- Les PDFs "numeriques" (non scannes) contiennent deja du texte extractible : toujours verifier avant de lancer l'OCR

### Post-processing

Le texte brut de l'OCR contient des erreurs qu'il faut corriger :
- Correction orthographique contextuelle avec dictionnaire metier
- Reconstruction des tableaux (detection de lignes/colonnes via des modeles layout-aware comme LayoutLMv3 ou Table Transformer)
- Preservation de l'ordre de lecture (multi-colonnes, headers/footers)

## 2. Le NER : du texte brut aux entites exploitables

Le NER (Named Entity Recognition) est la deuxieme brique : identifier et extraire les entites utiles dans le texte OCR.

### Les approches

- **NER classique (CRF/BiLSTM)** : rapide, peu de donnees necessaires, mais limitee aux patterns simples et reguliers.
- **NER Transformer (BERT/CamemBERT fine-tune)** : meilleure comprehension du contexte, gere les entites ambigues. Necessite 500+ exemples annotes, mais le gain en precision est massif. Mon choix pour la production en francais.
- **LLM en zero-shot/few-shot** : pas de donnees d'entrainement necessaires, flexible sur les schemas d'extraction. Plus lent et moins precis sur les entites tres specifiques, mais imbattable pour le prototypage.

### Entites typiques par domaine

**Factures** : numero de facture, date d'emission, montant HT/TTC, TVA, IBAN, nom du fournisseur, adresse de facturation.

**Contrats** : parties contractantes, dates de debut/fin, montant, clauses speciales, conditions de resiliation.

**Pieces d'identite** : nom, prenom, date de naissance, numero de document, date d'expiration, nationalite. C'est ce que j'ai automatise chez SOMA pour la validation de documents.

**Formulaires medicaux** : nom du patient, numero de securite sociale, diagnostic, traitements prescrits.

### L'annotation : le goulot d'etranglement

Pas de NER performant sans donnees annotees de qualite. Outils :
- **Label Studio** : open source, interface web, multi-annotateurs, le plus complet
- **Prodigy** (Explosion AI) : annotation avec active learning, tres efficace pour reduire le volume necessaire
- **Doccano** : simple, adapte aux projets NER/classification de petite taille

## 3. L'architecture du pipeline

```
Documents (PDF/Image, par milliers)
    |
    v
[Triage]               -->  PDF natif ? Extraire le texte directement.
    |                        PDF scanne ? Envoyer a l'OCR.
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
[NER / LLM Extract]   -->  CamemBERT fine-tune / LLM JSON mode
    |
    v
[Validation]           -->  Regles metier, score de confiance, seuils
    |
    v
[Sortie Structuree]    -->  JSON / PostgreSQL / API
    |
    v
[Feedback Loop]        -->  Corrections humaines reinjectees dans le training set
```

## 4. Metriques et evaluation

### OCR
- **CER (Character Error Rate)** : pourcentage de caracteres mal reconnus. Cible : <2% sur documents propres, <5% sur scans mediocres.
- **WER (Word Error Rate)** : pourcentage de mots mal reconnus. Cible : <5%.

### NER
- **Precision** : parmi les entites detectees, combien sont correctes ?
- **Recall** : parmi les entites reelles, combien ont ete detectees ?
- **F1-score** : moyenne harmonique. Cible : >90% par type d'entite en production.
- **Exact match vs partial match** : un montant "1 234,56" detecte comme "1 234" est un partial match. En production, seul l'exact match compte.

### Pipeline end-to-end
- **Extraction Accuracy** : pourcentage de champs correctement extraits par document. C'est LA metrique business.
- **Throughput** : documents traites par minute. Critique pour le dimensionnement.
- **Taux de rejet** : pourcentage de documents envoyes en review humain (score de confiance trop bas).

## 5. Les petits LLMs changent la donne

L'arrivee de petits LLMs performants (3-8B parametres) ouvre une alternative puissante au pipeline NER classique.

### Approche VLM (Vision Language Model)

Au lieu de faire OCR puis NER separement, un VLM analyse directement l'image et extrait les informations en une seule passe :

- **MiniCPM-V 2.6** (8B) : comprend les layouts complexes, tableaux, diagrammes. INT4 quantifie = ~4 Go RAM.
- **Qwen2-VL** (7B) : excellent sur les documents multilingues, resolutions arbitraires.
- **Florence-2** (0.7B) : ultra-leger, specialise OCR et detection d'objets.

L'avantage est enorme : pas de pipeline multi-etapes, le VLM gere OCR + comprehension + extraction en un seul appel. Moins de code, moins de bugs, moins de maintenance.

### LLM pour l'extraction structuree (JSON mode)

Apres l'OCR, au lieu d'un modele NER dedie, utiliser un petit LLM en mode JSON :

```
System: Extrais les informations suivantes du texte OCR au format JSON:
{"numero_facture": "", "date": "", "montant_ttc": "", "fournisseur": ""}

User: [texte OCR brut de la facture]
```

Modeles adaptes :
- **Phi-4-mini (3.8B)** : rapide sur CPU, bon suivi d'instructions, JSON fiable
- **Qwen 2.5 7B** : excellent multilingual, structured output
- **Llama 3.2 3B** : tres rapide, suffisant pour l'extraction simple

### Creation de datasets : le LLM comme annotateur

C'est la ou ca devient vraiment puissant. Au lieu d'annoter manuellement des milliers de documents :

1. Passer un echantillon (100-200 docs) dans un gros LLM (GPT-4, Claude) avec un prompt d'extraction detaille
2. Verifier manuellement les resultats, corriger les erreurs
3. Utiliser ce dataset comme base d'entrainement pour un modele NER dedie ou un petit LLM fine-tune
4. Le modele fine-tune traite les millions de documents restants a haute vitesse

C'est de la creation de dataset pure et dure : le gros LLM "devine" les annotations, l'humain valide, et le petit modele scale. On passe de semaines d'annotation manuelle a quelques jours.

### Quand utiliser quelle approche ?

- **OCR + NER classique** : documents standardises avec layout fixe, volume eleve (>1000/jour), latence critique (<1s)
- **OCR + petit LLM** : schemas d'extraction variables, besoin de flexibilite, volume modere
- **VLM direct** : documents avec layouts complexes (tableaux, graphiques), quand la qualite prime sur la latence
- **LLM-as-annotator + NER fine-tune** : quand vous avez des millions de documents et peu de donnees annotees

## 6. Production : ce qui fait la difference

### Scoring de confiance

Chaque extraction a un score de confiance. En dessous du seuil (typiquement 0.85), le document part en review humain. C'est non-negociable en production : mieux vaut ralentir que d'injecter des donnees fausses dans le systeme.

### Feedback loop

Les corrections humaines sont reinjectees dans le dataset d'entrainement. Le modele s'ameliore en continu. Apres 3 mois de production, notre F1 est passe de 0.88 a 0.94 uniquement grace au feedback loop.

### Scaling

Pour traiter des milliers de documents par heure :
- Paralleliser l'OCR (CPU-bound, scale lineaire avec les cores)
- Batch les appels NER/LLM
- Queue (Redis/Kafka) pour decoupler l'ingestion du traitement
- Monitoring du throughput et du taux de rejet en temps reel

### Les cas limites sont la norme

En production, 30% des documents sont des cas limites : documents tournes, multi-pages avec layouts differents, tampons sur le texte, ecriture manuscrite, qualite de scan degradee. Le pipeline doit avoir des fallbacks clairs : VLM pour les cas complexes, review humain pour les cas critiques.

## Conclusion

Transformer des millions de PDFs non structures en donnees structurees est un probleme d'ingenierie, pas de recherche. Les briques sont la : PaddleOCR pour le texte, CamemBERT pour le NER, les petits LLMs pour la flexibilite, les VLMs pour les cas complexes.

Ce qui fait la difference entre un prototype et un systeme de production, c'est le preprocessing, le scoring de confiance, le feedback loop, et la capacite a gerer les 30% de cas limites sans faire crasher le pipeline.

*Cet article s'appuie largement sur mon experience chez SOMA, ou j'ai automatise la validation de documents d'identite avec des pipelines OCR + NER. Traiter des milliers de documents par jour m'a appris que les 30% de cas limites ne sont pas des exceptions - c'est le vrai probleme a resoudre.*

*Michail Berjaoui - Janvier 2025*
