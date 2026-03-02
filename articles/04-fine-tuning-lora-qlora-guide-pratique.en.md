---
title: "LoRA/QLoRA Fine-tuning: A Practical Guide to Adapting an LLM to Your Domain"
date: 2024-12-05
tags: Fine-tuning, LoRA, QLoRA, PEFT, Unsloth, Dataset
summary: "Step-by-step guide to fine-tuning an open source LLM with LoRA and QLoRA: dataset preparation, hyperparameter configuration, training, evaluation, and adapter deployment."
---

# LoRA/QLoRA Fine-tuning: A Practical Guide to Adapting an LLM to Your Domain

## Introduction

Prompting has its limits. When an LLM doesn't produce expected results despite well-crafted prompts, fine-tuning becomes necessary. Thanks to LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA), it is now possible to adapt a 7B to 70B parameter model on a single consumer GPU (24 GB VRAM).

This article is a practical guide drawn from several fine-tuning runs performed in production, covering data preparation through deployment.

## 1. When to Fine-tune (and When Not To)

### Fine-tuning is justified when:

- The model doesn't master your **domain terminology** (medical, legal, financial)
- You need a **strict output format** (specific JSON, structured reports)
- **Few-shot prompting** doesn't yield consistent results
- You want to reduce **latency** by eliminating long system prompts

### Fine-tuning is NOT the solution when:

- You don't have **at least 500 quality examples**
- The problem comes from poor **retrieval** in your RAG pipeline
- Better **prompt engineering** could suffice
- You don't have defined **evaluation metrics**

## 2. Dataset Preparation

Dataset quality is the #1 success factor. Garbage in, garbage out.

### Data Format

The standard format is a list of conversations in JSON:

```json
{
  "conversations": [
    {"role": "system", "content": "You are a legal assistant specializing in French labor law."},
    {"role": "user", "content": "What are the notice periods for a 3-year permanent contract (CDI)?"},
    {"role": "assistant", "content": "For an employee on a permanent contract (CDI) with 3 years of seniority, the legal notice period is 2 months..."}
  ]
}
```

### Dataset Size

- **Minimum viable**: 500 examples for a style/format change
- **Recommended**: 1,000-5,000 examples for domain adaptation
- **Optimal**: 5,000-10,000 examples for deep behavioral change

### Critical Cleaning

- Remove duplicates and near-duplicates
- Verify response consistency
- Balance categories if doing classification
- Manually validate a 10% sample

## 3. LoRA Configuration

### Key Hyperparameters

- **Rank (r)**: Dimension of adaptation matrices. r=8 for style, r=32-64 for domain. The higher the rank, the more the model can learn, but the greater the overfitting risk.
- **Alpha**: Scaling factor. Rule of thumb: alpha = 2 x rank.
- **Target modules**: Which layers to adapt. By default, attention layers (q_proj, v_proj). For more capacity, add k_proj, o_proj, gate_proj, up_proj, down_proj.
- **Dropout**: 0.05-0.1 to prevent overfitting.

### QLoRA Configuration

QLoRA loads the base model in 4-bit (NF4) and only trains LoRA adapters in FP16/BF16. This divides VRAM by 4:

- Llama 3.1 8B: ~6 GB VRAM (vs 16 GB in FP16)
- Llama 3.1 70B: ~40 GB VRAM (vs 140 GB in FP16)

## 4. Training

### With Unsloth (recommended, 2x faster)

Unsloth automatically optimizes CUDA kernels for LoRA/QLoRA. It supports Llama, Mistral, Phi, Qwen, Gemma.

### Training Parameters

- **Learning rate**: 2e-4 to 5e-5 (lower for larger models)
- **Epochs**: 1-3 (rarely more, risk of catastrophic forgetting)
- **Batch size**: As large as possible within VRAM (gradient accumulation if needed)
- **Warmup**: 5-10% of steps
- **Scheduler**: Cosine decay

### Warning Signs During Training

- **Loss not decreasing**: Learning rate too low, or rank too small
- **Loss decreasing too fast**: Overfitting, reduce epochs
- **Loss oscillating**: Learning rate too high

## 5. Fine-tuning for RAG

An increasingly common use case: fine-tuning an LLM to better leverage context retrieved by a RAG pipeline.

### Why Fine-tune for RAG?

Even the best out-of-the-box LLMs have limitations in RAG:
- They can **ignore the provided context** and answer from memory (hallucination)
- They don't always know how to **cite their sources** reliably
- The citation format can be **inconsistent** (sometimes footnotes, sometimes inline)
- They can **fabricate passages** that aren't in the context

### Dataset for RAG Fine-tuning

The dataset must teach the model to:
1. **Always base its answer on the provided context** (and say so when the context doesn't contain the answer)
2. **Cite sources** with a consistent format (`[Source: name, p.X]`)
3. **Synthesize** multiple passages rather than copy-paste

Typical format:

```json
{
  "conversations": [
    {"role": "system", "content": "Answer based solely on the provided context. Cite your sources."},
    {"role": "user", "content": "Context:\n[Source: report.pdf, p.12] Revenue increased by 15%...\n[Source: balance.pdf, p.3] Net margin is 8.2%...\n\nQuestion: What is the company's financial health?"},
    {"role": "assistant", "content": "The company shows good financial health: revenue grew by 15% [Source: report.pdf, p.12] with a net margin of 8.2% [Source: balance.pdf, p.3]."}
  ]
}
```

### Synthetic Data Generation

When you don't have enough question-answer pairs, use a more powerful LLM (GPT-4, Claude) to generate them:

1. Take your real chunked documents
2. For each chunk, ask the LLM to generate 3-5 questions that the chunk answers
3. Ask it to generate answers with citations
4. Manually validate a 10-20% sample

This approach (distillation) lets you go from 100 manual examples to 2000+ quality examples.

## 6. Evaluation

Never evaluate solely on training loss. Metrics to track:

- **Perplexity** on a held-out test set (measures model "surprise")
- **Human evaluation**: sample of 50-100 responses manually scored on specific criteria (relevance, format, factuality)
- **LLM-as-judge**: use GPT-4 or Claude to score responses on defined rubrics. More scalable than human evaluation, good correlation.
- **Task-specific metrics**: accuracy for classification, BLEU/ROUGE for generation, F1 for NER
- **RAGAS** (for RAG): faithfulness, answer relevancy, context precision, citation correctness
- **Regression check**: verify the model hasn't lost its general capabilities (test on MMLU, HellaSwag)

### A/B Testing in Production

The ultimate test: deploy the fine-tuned model in parallel with the base model and compare user metrics (thumbs up/down, reformulations, time spent on the response).

## 7. Post-Fine-tuning Alignment (DPO)

After SFT (Supervised Fine-Tuning), you can go further with **DPO (Direct Preference Optimization)**: provide pairs (preferred response, rejected response) so the model learns your quality preferences. DPO is simpler than RLHF (no need for a separate reward model) and works well with HuggingFace's TRL.

## 8. Deployment

### Adapter Merging

Merge the LoRA adapter into the base model to eliminate inference overhead:

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")
```

### Post-Merge Quantization

Convert the merged model to GGUF for deployment with llama.cpp:

```bash
python convert_hf_to_gguf.py ./merged-model --outtype f16
./llama-quantize merged-model-f16.gguf merged-model-Q4_K_M.gguf Q4_K_M
```

The resulting GGUF model is directly usable in llama.cpp, llama-cpp-python, or Ollama.

### Stackable Adapters

A major advantage of LoRA: you can maintain multiple adapters for the same base model. A "legal" adapter, a "medical" one, a "RAG-optimized" one. Routing is done at the application level; the base model is loaded only once in memory.

## Conclusion

Fine-tuning with LoRA/QLoRA democratizes LLM adaptation. The keys to success: a quality dataset (500+ examples minimum), rigorous evaluation (never just the loss), and an iterative approach. Start with a small dataset and low rank, measure with concrete metrics, then gradually increase. For RAG, fine-tuning significantly improves the model's ability to use context and cite its sources.

*In MIRROR, the base Llama 3.1 8B handles RAG citations reasonably well out of the box thanks to careful prompt engineering (see the RAG article). Fine-tuning would be the next step if citation quality needed improvement - the RAG dataset format described above is exactly how I'd approach it.*

*Michail Berjaoui - December 2024*
