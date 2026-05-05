# Multi-GPU EP=2 Optimization Opportunities

## Current State (2026-03-04)

- 48 NCCL all_reduce calls per decode step (one per layer — all 48 layers have MoE)
- Each payload is 4 KB (2048 hidden_size x 2 bytes BF16) — pathologically small for NCCL
- All 48 are serialized in the CUDA graph (sync all_reduce on compute stream)
- Total comm volume per step: only 192 KB, but per-call latency dominates
- Current throughput: 100-112 tok/s EP=2 (vs 106 tok/s single-GPU K=2)

---

## 1. Custom Small-Message All-Reduce (HIGH IMPACT)

For 2 ranks with 4 KB payloads, NCCL is massive overkill. A custom kernel that does a
direct RDMA write + local add could cut per-call latency from ~20-50us to ~5us.
For 48 calls, that's 0.7-2.2ms saved per step.

**Approaches:**
- **NVSHMEM** peer-to-peer put/get
- **GPUDirect RDMA** with a persistent kernel

**Research:** NVRAR (arXiv 2511.09557) achieves 1.9-3.6x lower latency than NCCL for
small messages using recursive doubling with NVSHMEM.

## 2. NCCL 2.27 Symmetric Memory (HIGH IMPACT, easy)

NCCL 2.27 introduced symmetric memory buffers at identical virtual addresses — up to
9x latency reduction for small messages. Uses `ncclCommWindowRegister()`.

Need to check if it works over RoCE (not just NVLink).

Source: https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/

## 3. Coalesced All-Reduce Kernel (MEDIUM-HIGH IMPACT)

Replace 48 separate NCCL calls with a single persistent kernel that handles all 48
reductions. Total 192 KB fits in a single RDMA transfer (~62us at 3.1 GB/s).
Eliminates 47 kernel launch overheads.

**Challenge:** Layer N+1's input depends on layer N's all_reduce result, so you can't
batch them all at once. But a persistent kernel that stays resident and handles each
all_reduce via flag-signaling instead of kernel launch/teardown could help.

## 4. Communication-Computation Overlap (MEDIUM IMPACT, architectural)

Currently impossible because `graph_capture=true` forces sync all_reduce. Options:
- Use multi-stream CUDA graphs (capture across compute + comm streams)
- Abandon graphs for EP and use async all_reduce with overlap

Layer dependency (N+1 reads N's all_reduce result) makes pure overlap hard, but
attention layers provide natural windows: layer N's all_reduce could overlap with
layer N+1's pre-attention norm + QKV projection (~7 kernels before MoE).

**Research:** ScMoE (arXiv 2404.05019) achieves 100% communication overlap via
shortcut connections (requires retraining). DeepEP hook-based overlap (DeepSeek, 2025).

## 5. FP16 to FP8 All-Reduce (LOW IMPACT)

Halve payload from 4 KB to 2 KB. At these tiny sizes, barely matters — latency is
launch overhead, not bandwidth.

## 6. Expert Co-Activation Placement (MEDIUM IMPACT, model-specific)

Analyze which expert pairs are co-activated. If both selected experts are on the same
rank, that rank's all_reduce contribution is a no-op — could skip.

**Research:** Occult (arXiv 2505.13345) achieves >1.5x speedup through expert placement
optimization + customized sparse MatMul kernels.

## 7. Grace CPU as RDMA Proxy (SPECULATIVE)

UCCL-EP (arXiv 2512.19849) uses CPU-managed RDMA instead of GPU-initiated NCCL.
The Grace CPU has direct unified memory access. Could issue RoCE transfers without
GPU SM involvement. Achieved up to 2.1x dispatch/combine throughput over DeepEP.

---

## References

- NVRAR: https://arxiv.org/abs/2511.09557
- NCCL 2.27: https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/
- ScMoE: https://arxiv.org/abs/2404.05019
- DeepEP: https://github.com/deepseek-ai/DeepEP
- MegaScale-MoE: https://arxiv.org/abs/2505.11432
- MegaScale-Infer: https://arxiv.org/abs/2504.02263
- Occult: https://arxiv.org/abs/2505.13345
- UCCL-EP: https://arxiv.org/abs/2512.19849
- FUSCO: https://arxiv.org/abs/2512.22036
- FinDEP: https://arxiv.org/abs/2512.21487
- MoE-Spec: https://arxiv.org/abs/2602.16052
- HD-MoE: https://arxiv.org/abs/2509.09420
- NVIDIA Low-Latency Comm: https://developer.nvidia.com/blog/optimizing-for-low-latency-communication-in-inference-workloads-with-jax-and-xla/
