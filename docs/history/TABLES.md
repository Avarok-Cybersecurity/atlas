# Atlas Demo Results

> **Model**: Qwen3-Next-80B-A3B-Instruct-NVFP4 (80B params, 3B active/token)
> **Hardware**: Single NVIDIA GB10 GPU (SM121, 120 GB LPDDR5X @ 273 GB/s)

**Why this matters:**
- Same model, same GPU, same quantization across all three frameworks
- Atlas is pure Rust + CUDA -- zero Python, zero PyTorch, zero framework overhead
- Every kernel is hand-tuned for this exact (hardware, model) pair
- Weights load in ~55 seconds vs ~7 minutes for vLLM -- PTX is pre-embedded at build time

---

## Single-Request Decode Speed

| Framework | Image | Throughput | vs NVIDIA |
|---|---|---:|---:|
| **Atlas Spark** | `atlas-spark` | **82 tok/s** | **2.8x** |
| Avarok vLLM (custom NVFP4 kernel) | `avarok/dgx-vllm-nvfp4-kernel:v22` | 36.4 tok/s | 1.2x |
| vLLM (FP8, stock image) | `dgx-vllm:latest` | ~34 tok/s | 1.1x |
| NVIDIA TensorRT-LLM | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc2` | 29.6 tok/s | 1.0x |

---

## The Journey: 3.6 to 99 tok/s (27.5x)

- **3.6 -> 41 tok/s** -- Decode-specialized GEMV kernels, batched MoE dispatch, CUDA graphs
- **41 -> 80 tok/s** -- NVFP4 dequant breakthrough (shared memory E2M1 LUT alone was +71%)
- **80 -> 99 tok/s** -- Kernel fusion campaign across all 48 layers (9 fusions, 20 CUDA kernels total)

---

## Source Code to First Token

| Phase | Atlas Spark | vLLM (NVFP4) |
|---|---:|---:|
| Build from source (`docker build`) | ~60 s | ~30-45 min |
| Weight loading (47 GB) | ~55 s | ~4 min |
| Init (KV cache, graphs, JIT) | < 1 s | ~3 min |
| **Total: source to first token** | **~2 min** | **~40+ min** |

---

## Kernel Highlights (32/32 wins vs PyTorch)

| Kernel | Speedup |
|---|---:|
| RoPE (GQA 16:2) | **18.2x** |
| Conv1d decode (SSM layers) | **8.9x** |
| RMSNorm | **9.3x** |
| Gated Delta Rule (SSM prefill) | **7.9x** |
| Decode Attention (paged KV) | **6.0x** |
| MoE W4A16 (256 experts, top-10) | **3.9x** |

---

## Workload Benchmarks (Atlas vs Avarok vLLM v22, no MTP)

Same model, same hardware, same quantization (NVFP4). No speculative decoding on either side. Raw kernel speed.

| Workload | ISL | OSL | Atlas (tok/s) | vLLM (tok/s) | Atlas Advantage |
|---|---:|---:|---:|---:|---:|
| Short chat | 256 | 256 | **80.2** | 44.0 | **1.8x** |
| Code generation | 128 | 1024 | **79.4** | 41.7 | **1.9x** |
| Summarization | 1024 | 128 | **78.3** | 47.0 | **1.7x** |
| Standard chat | 1024 | 1024 | **75.5** | 41.8 | **1.8x** |
| Long reasoning | 1024 | 8192 | **69.6** | 40.9 | **1.7x** |
| RAG / document QA | 8192 | 1024 | **55.1** | 40.9 | **1.3x** |

> Atlas wins every workload. **1.3x to 1.9x faster** across the board -- no speculation, just faster kernels.