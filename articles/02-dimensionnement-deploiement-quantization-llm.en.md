---
title: "Practical & Strategic Guide: LLM Sizing, Deployment and Quantization"
date: 2025-02-20
tags: Quantization, VRAM, GPU, Inference, Production, LLM, Deployment
summary: "Architectural standards and best practices for deploying LLMs in production. Inference optimization via quantization, VRAM sizing on GPUs (H100, A100, RTX), inference engine selection (TRT-LLM, vLLM, llama.cpp), and deployment decision tree."
---

# Practical & Strategic Guide (2025): LLM Sizing, Deployment and Quantization

**Status:** Validated | **Audience:** MLOps Engineers, AI Architects, DevOps

This document covers architectural standards and best practices for deploying Large Language Models (LLMs) in production environments. It focuses on inference optimization via quantization, VRAM sizing on GPUs (NVIDIA H100, A100, RTX), and inference engine selection (TRT-LLM, vLLM, llama.cpp).

*For MIRROR, I went with the "CPU Only / Edge" path: llama.cpp with GGUF Q6_K on a 64 GB Hetzner server. The decision tree below is the framework I used to arrive at that choice.*

## 1. Foundations: LLM Memory Management

To master quantization, it is imperative to understand how a model's memory footprint is distributed during inference. VRAM usage breaks down into three pillars:

- **Weights**: the model parameters learned during training. They dictate the minimum size on disk and in memory.
- **Activations**: intermediate results computed at each inference step. Their footprint is generally minimal.
- **KV-Cache (Key-Value Cache)**: key and value tensors from the attention mechanism, preserved throughout generation to avoid recomputing history. It grows linearly with context length and the number of concurrent requests.

### The KV-Cache Formula (with GQA)

The memory allocated to the KV-cache is estimated as:

```
KV_bytes ≈ batch_size × sequence_length × num_layers × (2 × hidden_size) × (bytes_per_dtype / gqa_factor)
```

Concrete example: for Llama 3 70B in FP16, with a batch of 4 and 8192 tokens of context, the KV-cache consumes approximately 20 GB of VRAM on top of the model weights.

### The PagedAttention Optimization

Regardless of numerical format, **PagedAttention** (introduced by vLLM) optimizes KV-cache usage by splitting it into non-contiguous pages. This reduces memory waste due to fragmentation from 60-80% to less than 4%, enabling maximum in-flight batching.

## 2. VRAM Sizing: How Much Do We Actually Save?

The golden rule of sizing relies on converting parameter count to bytes:

- **FP16 (16 bits)**: 1 parameter = 2 bytes (x2 multiplier)
- **FP8 / INT8 (8 bits)**: 1 parameter = 1 byte (x1 multiplier)
- **INT4 (4 bits)**: 1 parameter = 0.5 bytes (x0.5 multiplier)

*Note: always add a 15% to 20% margin for KV-cache and activations.*

### Case Study on Standard GPUs

| Target Model | Precision | Weight VRAM | Total VRAM (with context) | Minimum Recommended GPU | Savings vs FP16 |
|:---|:---|:---|:---|:---|:---|
| **8B** (Llama 3 8B) | FP16 (Baseline) | ~16 GB | ~18-20 GB | 1x RTX 3090/4090 (24 GB) | Ref. |
| **8B** | FP8 / INT8 | ~8 GB | ~10-12 GB | 1x RTX 4060 Ti (16 GB) or T4 | **-50%** |
| **8B** | INT4 (AWQ/GGUF) | ~4 GB | ~6-8 GB | 1x RTX 3060 (12 GB) | **-75%** |
| **70B** (Llama 3 70B) | FP16 (Baseline) | ~140 GB | ~160 GB | 2x H100 (80 GB) | Ref. |
| **70B** | FP8 / INT8 | ~70 GB | ~78 GB | **1x H100 (80 GB)** | **-50%** |
| **70B** | INT4 (AWQ/GGUF) | ~35 GB | ~42 GB | 1x A6000 (48 GB) | **-75%** |

### The Trade-off: VRAM vs Speed vs Quality

- **Saving 50% VRAM (FP8/INT8)**: quality loss (perplexity) is nearly undetectable (< 1%). On an H100, FP8 doubles inference speed thanks to specialized Tensor Cores. This is a pure win.
- **Saving 75% VRAM (INT4)**: qualitative degradation becomes measurable (lost stylistic nuances, slight hallucinations on complex code/math). Speed gains are moderate because 4-bit computation isn't always natively accelerated.

## 3. Key Quantization Methods

Several Post-Training Quantization (PTQ) approaches dominate the ecosystem:

- **SmoothQuant (INT8 W8A8)**: smooths extreme activation values (outliers) by transferring part of their amplitude to the weights via rescaling. Enables stable 8-bit inference without retraining, ideal on Ampere/Hopper architectures. (Xiao et al., ICML 2023)

- **AWQ (Activation-aware Weight Quantization, INT4)**: identifies and protects the ~1% most critical weights (stored in FP16), and quantizes the remaining 99% to 4 bits. Offers the best quality/compression ratio for VRAM-constrained environments. (Lin et al., MLSys 2024)

- **GPTQ (INT3/INT4)**: one-shot method using Hessian approximation to compensate for quantization error block by block. Very popular in the open-source community. (Frantar et al., ICLR 2023)

- **GGUF (llama.cpp)**: granular quantization format with numerous variants (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0). Each variant offers a different trade-off between size and quality. Q4_K_M is the sweet spot for most CPU use cases.

## 4. Inference Engine Comparison (Serving)

### A. TensorRT-LLM (NVIDIA)
Ultra-optimized compiler and runtime specific to NVIDIA GPUs.
- **Advantages**: native FP8 support on H100, W8A8, AWQ, in-flight batching. Delivers maximum throughput and absolute minimum latency (up to 4.6x more throughput on H100 vs A100).
- **Disadvantages**: hardware lock-in (CUDA only), requires prior compilation (engine build).

### B. vLLM (Open-Source)
Python/C++ server developed by UC Berkeley, industry standard.
- **Advantages**: native PagedAttention, dynamic FP8/INT8 support, on-the-fly loading. OpenAI-compatible Python API.
- **Disadvantages**: pure single-request latency slightly higher than TRT-LLM, although throughput under heavy load is exceptional.

### C. llama.cpp / GGUF (CPU & Edge)
Minimalist C/C++ implementation for local execution.
- **Advantages**: hardware-agnostic (CPU, Apple Silicon, small GPUs). Supports highly granular quantization formats (Q4_K_M, Q8_0). SIMD acceleration (AVX2/AVX-512).
- **Disadvantages**: doesn't fully leverage industrial Tensor Cores. Not recommended for high-load data center servers.

## 5. Decision Tree: Which Strategy to Deploy?

1. **H100 End-to-End (Maximum Performance & Quality)**
  - Method: FP8 (W8A8) + KV-Cache FP8
  - Tool: TensorRT-LLM or vLLM (`--quantization fp8`)
  - Benefit: fits a 70B on a single H100 with speed and quality > 99% preserved

2. **The Universal Choice (A100 / Standard GPUs)**
  - Method: INT8 SmoothQuant (W8A8) + KV-Cache FP8
  - Tool: vLLM (`--quantization int8`)
  - Benefit: ideal when native FP8 is not supported

3. **High Density / Reduced Budget**
  - Method: INT4 AWQ (Weights only) + KV-Cache FP8
  - Tool: vLLM (`--quantization awq`)
  - Benefit: cuts VRAM costs by 4x. Perfect for internal chatbots

4. **CPU Only / Edge**
  - Method: GGUF Q4_K_M or Q3_K_M
  - Tool: llama.cpp / llama-cpp-python
  - Benefit: no GPU required, runs on laptop or CPU server (used in MIRROR)

## 6. Monitoring and SLIs in Production

| Target Metric | Alert Threshold (Example) | Corrective Action |
|:---|:---|:---|
| Latency (Time To First Token) | p99 > 500ms | Scale-out or switch to INT4 profile |
| KV-Cache Waste | > 8% | Check PagedAttention configuration |
| OOM Errors | > 0.5% of requests | Reduce max context or use FP8 KV |
| Throughput | < SLA (tokens/s) | Add replicas or switch to FP8 |

## 7. Compliance Reminder (Licensing)

Quantization is a technical transformation - it **does not change the source license of a model in any way**. A "merged" model systematically inherits the restrictions of all its original components. If commercial use or redistribution is prohibited on the base model, it remains prohibited on the GGUF or TensorRT engine.

## 8. Command Cheat-Sheet

```bash
# vLLM FP8 (H100)
vllm serve mistralai/Mistral-7B-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 16384

# vLLM INT8 (A100 / standard GPUs)
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

## References

- **SmoothQuant**: Xiao et al. (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs*. ICML
- **AWQ**: Lin et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression*. MLSys
- **GPTQ**: Frantar et al. (2023). *GPTQ: Accurate Post-Training Quantization for GPT*. ICLR
- **vLLM & PagedAttention**: Kwon et al. (2023). *Efficient Memory Management for LLM Serving with PagedAttention*. SOSP
- **LLM.int8()**: Dettmers et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS

*Full article available on [GitHub](https://github.com/SoMika00/Quant_llm)*

*Michail Berjaoui - February 2025*
