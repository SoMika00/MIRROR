---
title: "From Millions of PDFs to Structured Data: OCR + NER Pipeline in Production"
date: 2025-01-15
tags: OCR, NER, Documents, Pipeline, Production, VLM, Small LLM, Dataset Creation
summary: "How to transform millions of unstructured documents (scanned PDFs, invoices, contracts) into exploitable structured data. OCR, NER, VLMs, dataset creation, and the real challenges of production."
---

# From Millions of PDFs to Structured Data: OCR + NER Pipeline in Production

## The Problem

Every company has the same problem: millions of scanned documents (invoices, contracts, forms, identity documents) containing critical information, but trapped in pixels. Entire teams spend their days copying numbers into spreadsheets. It's slow, expensive, and riddled with human errors.

The promise: take a scanned PDF, automatically extract useful fields (amount, date, name, IBAN), and inject them directly into a database or ERP. At scale, that's thousands of hours of manual data entry saved.

The reality: it's one of the most underestimated problems in data engineering. Scan quality is catastrophic, layouts are unpredictable, and edge cases represent 30% of the volume.

## 1. The Power (and Limits) of OCR

OCR (Optical Character Recognition) is the first building block: transforming an image into raw text.

### The Engines

Three options depending on context:

- **Tesseract 5**: open source, mature, 100+ languages. Performant on clean documents with simple layouts. Free, but struggles with complex tables and poor quality scans.
- **PaddleOCR**: open source (Baidu), the best quality/price ratio. Excellent on complex layouts (tables, columns, forms). Detection + recognition in a single pipeline. My default choice for production.
- **DocTR** (Mindee): open source, Transformer-based. Very good on European documents. Heavier than PaddleOCR but more accurate on difficult cases.
- **Google Document AI / Azure Form Recognizer**: cloud solutions, best raw performance, but per-page cost and cloud dependency. Worth considering for massive volumes with budget.

### Preprocessing: 80% of the Work

OCR quality depends directly on image quality. Good preprocessing transforms a mediocre OCR into an excellent one:

- **Deskew**: correct tilt. A document rotated by 2 degrees is enough to drop accuracy by 15%.
- **Adaptive binarization**: convert to black and white with a local threshold. Scans with uneven lighting need this.
- **Denoising**: remove stains, artifacts, stamps overlapping text.
- **Resolution**: upscale anything below 300 DPI. Below that, small characters become unreadable for OCR.

### The Real Challenges

What tutorials don't tell you:
- Stamps and signatures over text destroy detection
- Tables without visible borders are a nightmare (the engine mixes up columns)
- Handwriting remains an open problem for classical OCR
- "Digital" PDFs (not scanned) already contain extractable text: always check before running OCR

### Post-processing

Raw OCR text contains errors that must be corrected:
- Contextual spell-checking with domain-specific dictionary
- Table reconstruction (line/column detection via layout-aware models like LayoutLMv3 or Table Transformer)
- Preserving reading order (multi-column, headers/footers)

## 2. NER: From Raw Text to Exploitable Entities

NER (Named Entity Recognition) is the second building block: identifying and extracting useful entities from OCR text.

### The Approaches

- **Classical NER (CRF/BiLSTM)**: fast, requires little data, but limited to simple and regular patterns.
- **Transformer NER (BERT/CamemBERT fine-tuned)**: better contextual understanding, handles ambiguous entities. Requires 500+ annotated examples, but the precision gain is massive. My choice for production in French.
- **LLM in zero-shot/few-shot**: no training data needed, flexible on extraction schemas. Slower and less precise on very specific entities, but unbeatable for prototyping.

### Typical Entities by Domain

**Invoices**: invoice number, issue date, amount excl. tax/incl. tax, VAT, IBAN, supplier name, billing address.

**Contracts**: contracting parties, start/end dates, amount, special clauses, termination conditions.

**Identity documents**: last name, first name, date of birth, document number, expiration date, nationality. This is what I automated at SOMA for document validation.

**Medical forms**: patient name, social security number, diagnosis, prescribed treatments.

### Annotation: The Bottleneck

No performant NER without quality annotated data. Tools:
- **Label Studio**: open source, web interface, multi-annotator, the most complete
- **Prodigy** (Explosion AI): annotation with active learning, very efficient for reducing required volume
- **Doccano**: simple, suited for small NER/classification projects

## 3. The Pipeline Architecture

```
Documents (PDF/Image, by the thousands)
    |
    v
[Triage]               -->  Native PDF? Extract text directly.
    |                        Scanned PDF? Send to OCR.
    v
[Image Preprocessing]  -->  Deskew, Binarization, Denoising, Upscale
    |
    v
[OCR Engine]           -->  PaddleOCR / DocTR / Tesseract
    |
    v
[Post-processing]      -->  Correction, Table reconstruction
    |
    v
[NER / LLM Extract]   -->  CamemBERT fine-tuned / LLM JSON mode
    |
    v
[Validation]           -->  Business rules, confidence score, thresholds
    |
    v
[Structured Output]    -->  JSON / PostgreSQL / API
    |
    v
[Feedback Loop]        -->  Human corrections reinjected into the training set
```

## 4. Metrics and Evaluation

### OCR
- **CER (Character Error Rate)**: percentage of misrecognized characters. Target: <2% on clean documents, <5% on poor scans.
- **WER (Word Error Rate)**: percentage of misrecognized words. Target: <5%.

### NER
- **Precision**: among detected entities, how many are correct?
- **Recall**: among actual entities, how many were detected?
- **F1-score**: harmonic mean. Target: >90% per entity type in production.
- **Exact match vs partial match**: an amount "1,234.56" detected as "1,234" is a partial match. In production, only exact match counts.

### End-to-End Pipeline
- **Extraction Accuracy**: percentage of correctly extracted fields per document. This is THE business metric.
- **Throughput**: documents processed per minute. Critical for sizing.
- **Rejection rate**: percentage of documents sent for human review (confidence score too low).

## 5. Small LLMs Are Changing the Game

The arrival of performant small LLMs (3-8B parameters) opens a powerful alternative to the classical NER pipeline.

### VLM Approach (Vision Language Model)

Instead of doing OCR then NER separately, a VLM directly analyzes the image and extracts information in a single pass:

- **MiniCPM-V 2.6** (8B): understands complex layouts, tables, diagrams. INT4 quantized = ~4 GB RAM.
- **Qwen2-VL** (7B): excellent on multilingual documents, arbitrary resolutions.
- **Florence-2** (0.7B): ultra-lightweight, specialized in OCR and object detection.

The advantage is enormous: no multi-step pipeline, the VLM handles OCR + comprehension + extraction in a single call. Less code, fewer bugs, less maintenance.

### LLM for Structured Extraction (JSON Mode)

After OCR, instead of a dedicated NER model, use a small LLM in JSON mode:

```
System: Extract the following information from the OCR text in JSON format:
{"invoice_number": "", "date": "", "total_amount": "", "supplier": ""}

User: [raw OCR text from the invoice]
```

Suitable models:
- **Phi-4-mini (3.8B)**: fast on CPU, good instruction following, reliable JSON
- **Qwen 2.5 7B**: excellent multilingual, structured output
- **Llama 3.2 3B**: very fast, sufficient for simple extraction

### Dataset Creation: The LLM as Annotator

This is where it gets truly powerful. Instead of manually annotating thousands of documents:

1. Run a sample (100-200 docs) through a large LLM (GPT-4, Claude) with a detailed extraction prompt
2. Manually verify the results, correct errors
3. Use this dataset as training data for a dedicated NER model or a small fine-tuned LLM
4. The fine-tuned model processes the remaining millions of documents at high speed

This is pure dataset creation: the large LLM "guesses" the annotations, the human validates, and the small model scales. You go from weeks of manual annotation to a few days.

### When to Use Which Approach?

- **OCR + classical NER**: standardized documents with fixed layout, high volume (>1000/day), critical latency (<1s)
- **OCR + small LLM**: variable extraction schemas, need for flexibility, moderate volume
- **Direct VLM**: documents with complex layouts (tables, graphics), when quality trumps latency
- **LLM-as-annotator + NER fine-tune**: when you have millions of documents and little annotated data

## 6. Production: What Makes the Difference

### Confidence Scoring

Every extraction has a confidence score. Below the threshold (typically 0.85), the document goes for human review. This is non-negotiable in production: better to slow down than to inject false data into the system.

### Feedback Loop

Human corrections are reinjected into the training dataset. The model improves continuously. After 3 months in production, our F1 went from 0.88 to 0.94 solely thanks to the feedback loop.

### Scaling

To process thousands of documents per hour:
- Parallelize OCR (CPU-bound, scales linearly with cores)
- Batch NER/LLM calls
- Queue (Redis/Kafka) to decouple ingestion from processing
- Real-time monitoring of throughput and rejection rate

### Edge Cases Are the Norm

In production, 30% of documents are edge cases: rotated documents, multi-page with different layouts, stamps over text, handwriting, degraded scan quality. The pipeline must have clear fallbacks: VLM for complex cases, human review for critical cases.

## Conclusion

Transforming millions of unstructured PDFs into structured data is an engineering problem, not a research one. The building blocks are there: PaddleOCR for text, CamemBERT for NER, small LLMs for flexibility, VLMs for complex cases.

What makes the difference between a prototype and a production system is preprocessing, confidence scoring, the feedback loop, and the ability to handle the 30% of edge cases without crashing the pipeline.

*This article draws heavily from my experience at SOMA, where I automated identity document validation using OCR + NER pipelines. Processing thousands of documents daily taught me that the 30% edge cases aren't exceptions - they're the real problem to solve.*

*Michail Berjaoui - January 2025*
