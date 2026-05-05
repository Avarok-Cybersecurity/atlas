# Atlas Spark — State-of-the-Art Research Report (v2)

**Research Date**: February 2026  |  **Target**: Multi-Spark (multi-node, 1 GB10 per node)

---

## Tier Summary

| Tier | Technique | Impact | Phase |
|---|---|---|---|
| 🔴 **T1** | Megakernel execution (Mirage MPK) | 1.2-6.7× latency | P2 |
| 🔴 **T1** | MTP speculative decoding (default) | ~2× decode | P3 |
| 🔴 **T1** | EAGLE-3 speculative decoding (optional) | ~6.5× decode | P3 |
| 🔴 **T1** | AllToAll overlap for multi-Spark MoE (Hybrid-EP / ScMoE) | Hide EP comm latency | P2 |
| 🟡 **T2** | KV cache compression (EvolKV) | 2-4× cache memory | P2 |
| 🟡 **T2** | Radix tree prefix caching | Avoid redundant prefill | P2 |
| 🟡 **T2** | Chunked prefill scheduling | Better TTFT | P2 |
| 🟡 **T2** | Disaggregated prefill/decode (multi-Spark) | Independent scaling | P3 |
| 🟢 **T3** | W4A8KV4 quantization (QServe) | Lower bandwidth | P3 |
| 🟢 **T3** | Automated warp specialization (Tawa) | Kernel optimization | P3+ |

---

## Tier 1: High-Impact

### 1. Megakernel Execution (Mirage MPK)

**Paper**: arXiv:2512.22219 (Dec 2025) — CMU, UW, Berkeley, NVIDIA, Tsinghua

Fuses entire LLM forward pass into a single CUDA kernel. On-GPU SM-level scheduling via t-Graph. **1.2-6.7× latency reduction.** Multi-GPU aware — overlaps computation with NCCL communication inside the megakernel.

**Atlas Spark fit**: Start with CUDA graphs (P2), evolve to persistent megakernel (P3). Our `GpuBackend` trait makes this swap transparent.

### 2. MTP Speculative Decoding (Default)

**Papers**: FastMTP (arXiv, Feb 2025), DeepSeek-R1, GLM-4.5

MTP trains additional prediction heads on the base model to draft K tokens per step. Qwen3-Next already has MTP support (2 tokens/step). **~2× decode speedup** with zero quality loss (lossless verification).

**Atlas Spark fit**: Default speculative strategy. MTP heads are small and already part of the model checkpoint — no extra model needed. CLI: `--speculative-method mtp --num-speculative-tokens 2`.

### 3. EAGLE-3 Speculative Decoding (Optional)

**Paper**: EAGLE-3 (2025) — 6.5× decode speedup via direct token prediction + multi-layer feature fusion.

Higher throughput than MTP but requires a separate draft head (trained MLP). Not all models ship with EAGLE heads.

**Atlas Spark fit**: Optional upgrade. CLI: `--speculative-method eagle3 --num-speculative-tokens 5`. Requires `--draft-head-path` pointing to the trained EAGLE-3 weights.

### 4. AllToAll Overlap for Multi-Spark MoE

**Papers**: Hybrid-EP (NVIDIA, 2025), ScMoE (arXiv, 2025), Occult (OpenReview, 2025)

Expert Parallelism AllToAll communication is the #1 bottleneck in multi-node MoE inference (can be 40-50% of runtime). Solutions:

| Method | Approach | Result |
|---|---|---|
| **Hybrid-EP** | Pipeline fine-grained chunks through NVLink→RDMA stages | Near hardware-limit BW |
| **ScMoE** | Shortcut-connected MoE, overlap computation with AllToAll | Up to 100% comm overlap |
| **Occult** | Co-locate co-activated experts to minimize cross-node sends | Up to 50% volume reduction |

**Atlas Spark fit**: Critical for multi-Spark. Implement in `spark-comm` as `CommBackend::all_to_all()` with pipelined chunking. Our `CommBackend` trait (SDD) allows swapping strategies transparently.

---

## Tier 2: Phase 2-3

### 5. KV Cache Compression (EvolKV)
Evolutionary per-layer budget search. Up to 4.38× compression. Implement as `KvCachePolicy` trait.

### 6. Radix Tree Prefix Caching
Reuse KV states for shared prefixes (system prompts, multi-turn). Add `PrefixCache` to `PagedKvCache`.

### 7. Chunked Prefill Scheduling
Split long prompts into chunks, interleave with decode. Part of continuous batching scheduler.

### 8. Disaggregated Prefill/Decode (Multi-Spark)
**Now relevant** with multi-Spark: assign prefill-heavy and decode-heavy DGX Sparks. Prefill is compute-bound, decode is memory-bound — different optimization profiles. Implement as scheduler policy.

---

## Tier 3: Future

### 9. W4A8KV4 Quantization (QServe)
4-bit weights, 8-bit activations, 4-bit KV. Requires new GEMM kernels.

### 10. Automated Warp Specialization (Tawa)
Compiler for producer/consumer warp patterns. Apply to decode attention kernel.

---

## Updated Roadmap

| Phase | Techniques | Single-Spark tok/s | Multi-Spark (2×) |
|---|---|---|---|
| **P1** | All Atlas kernels, paged KV, greedy | 42-45 | — |
| **P2** | CUDA graphs, fused kernels, batching, NCCL EP/TP | 50-55 | 90-100 |
| **P3** | MTP (default) + EAGLE-3 (optional) | 80-100 | 160-200 |
