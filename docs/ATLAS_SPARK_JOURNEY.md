# Atlas Spark: From Zero to 99 tok/s — The Full Story

## What Is Atlas Spark?

Atlas Spark is a **pure Rust LLM inference server** built entirely from scratch for the **Qwen3-Next-80B-A3B-Instruct-NVFP4** model running on a single **NVIDIA DGX Spark GB10** (Blackwell SM121, 119.7 GB LPDDR5X @ 273 GB/s).

There is no PyTorch, no Python, no vLLM in the hot path. Every operation — embedding lookup, attention with paged FP8 KV cache, Mamba SSM recurrent state updates, MoE expert routing across 512 experts, NVFP4 weight dequantization, RoPE, RMS norm, argmax — is implemented in Rust with hand-tuned CUDA kernels compiled to SM121 PTX.

**Final result**: **99.1 tok/s peak** (sustained ~96–97 tok/s), generating tokens from an 80-billion-parameter model on a single desktop GPU. For comparison, vLLM (the industry standard) achieves 36.4 tok/s on the same hardware without speculative decoding, and 59.9 tok/s with MTP speculative decoding.

---

## The Hardware

The DGX Spark GB10 is NVIDIA's Grace Blackwell Superchip for workstations:

- **GPU**: Blackwell GB10, Compute Capability 12.1 (SM121)
- **Memory**: 119.7 GB unified LPDDR5X at 273 GB/s peak bandwidth
- **Key constraint**: No HBM — memory bandwidth is the primary bottleneck for LLM inference at batch=1. Every optimization ultimately comes down to reading fewer bytes or reading them more efficiently.

SM121 is a brand-new architecture. Many existing CUDA libraries (FlashInfer fused MoE, TRT-LLM MoE backends, various CUTLASS kernels) crash or produce garbage output on it. This forced us to write every kernel from scratch.

---

## The Model

**Qwen3-Next-80B-A3B-Instruct-NVFP4** is an unusually complex architecture:

- **80 billion parameters** (MoE: ~3B active per token)
- **Hybrid**: 12 attention layers + 36 Mamba SSM (GDN) layers, interleaved
- **MoE**: 512 experts, top-10 routing per token, plus a shared expert per layer
- **NVFP4 quantization**: 4-bit E2M1 weights with FP8 E4M3 per-group scales
- **48 layers total**, each followed by an MoE feed-forward block

The hybrid attention + Mamba architecture is particularly challenging because the SSM layers maintain recurrent state that must be updated sequentially — no shortcut via KV caching like pure transformers.

---

## Phase 1: Atlas Kernel Library (Weeks 1–2)

The project started as a CUDA kernel library benchmarked against PyTorch on SM121. The goal was to prove that hand-tuned kernels could beat the existing ecosystem on this new hardware.

### What was built

- **Prefill attention**: Flash Attention v2 with `cp.async` prefetch, register-based softmax, 4-warp PV accumulation. Iterated through 47 versions (`inferspark_prefill_v3.cu` through `v47.cu`) to get it right.
- **Decode attention**: Paged attention with batched KV loads and tree-merge reduction across warps.
- **NVFP4 GEMM**: E2M1 LUT dequantization with FP8 block scales, 64×64 tiles with MMA tensor cores.
- **Dense BF16 GEMM**: 16×16 tile scalar GEMM for projection layers.
- **RMSNorm, RoPE, SiLU**: Standard transformer building blocks.
- **Gated Delta Rule (GDN)**: The Mamba SSM recurrent update, implemented as a fused CUDA kernel.
- **Causal Conv1d**: Sliding-window convolution for SSM preprocessing.

### Results

32 benchmarks against PyTorch, Atlas won 25/32:
- Prefill seq=256: **4.95x faster** (0.025ms vs 0.122ms)
- Decode seq=4096: **6.02x faster** (0.049ms vs 0.292ms)
- MoE batched GEMM: **3.87x faster** (8.4ms vs 32.6ms)
- Conv1d prefill: **9.95x faster** (0.053ms vs 0.530ms)
- GDN prefill: **7.89x faster** (1.4ms vs 11.1ms)

---

## Phase 2: vLLM Integration (Week 3)

With proven kernels, the next step was integrating into vLLM as an attention backend to see real-world impact.

### Key discoveries

- **Non-contiguous QKV views**: vLLM's QKV projection splits create non-contiguous tensors. The token-to-token stride is `(Hq + 2*Hkv) * D` instead of the expected `H * D`. Fixed by adding explicit stride parameters to all CUDA kernels.
- **FP8 KV cache stride mismatch**: vLLM allocates FP8 KV cache with BF16-sized memory, then views as uint8. This creates `key_cache.stride(0) = 2x` the contiguous stride. Writes went to wrong offsets, reads saw zeros. Fixed by passing `cache_stride` from Python.
- **Selective patching wins**: Patching individual ops (RMSNorm, SiLU, RoPE) with Atlas kernels *broke* torch.compile fusion boundaries. Removing 6 of 7 patches gave **+11% throughput** — only patch attention (the one op that's SM121-specific), let vLLM handle everything else.

### Result

**40.5 tok/s** with Atlas attention-only patching, FP8 KV cache, and CUDA graphs in FULL+PIECEWISE mode. Comparable to vLLM's 42 tok/s FlashInfer baseline.

### The realization

The vLLM approach had a ceiling. The Python overhead, the framework abstractions, the inability to fuse across layers — all of it limited what was possible. To go faster, we needed to own the entire pipeline.

---

## Phase 3: Building Atlas Spark from Scratch (Weeks 3–4)

Atlas Spark was a complete rewrite of the inference pipeline in pure Rust.

### Architecture

```
15 Rust crates:
├── atlas-core          Config, hardware detection, quantization metadata
├── atlas-kernels       29 PTX modules embedded at build time (compiled from .cu)
├── spark-runtime       CUDA driver API wrapper, KV cache, weight loader, buffer arena
├── spark-model         Model composition, layer traits, Qwen3 attention + SSM + MoE
├── spark-server        HTTP server, tokenizer, integration test
├── spark-comm          gRPC stub for future multi-GPU
└── 9 other support crates (atlas-py, atlas-quant, atlas-gemm, etc.)
```

### Design principles

1. **Zero-copy weight loading**: Safetensor files memory-mapped directly. NVFP4 weights used as-is. BF16 dense weights quantized to FP8 at load time (one-time cost).
2. **Architecture-agnostic model loop**: `TransformerLayer` trait with `decode()` method. Adding a new model means implementing the trait — the 48-layer loop stays unchanged.
3. **Buffer arena**: Pre-allocated scratch buffers sized once at init. Zero allocations during inference.
4. **Device-side everything**: Expert indices, routing weights, gate scalars — all computed on GPU. Zero host-device synchronization in the hot path.

### First result

The integration test passed: the model generated tokens correctly. But at **3.6 tok/s** (275 ms/token), it was 8x slower than vLLM. The theoretical floor was ~19 ms/token based on weight bandwidth. A 15x gap.

---

## Phase 4: The Optimization Journey (3.6 → 99.1 tok/s)

This is where the real work happened. Every commit brought a measurable speedup, each unlocked by understanding exactly where the bytes were going.

### Stage 1: Eliminating Gross Inefficiencies (3.6 → 41 tok/s, 11.4x)

| Commit | What Changed | Before | After | Gain |
|--------|-------------|--------|-------|------|
| `0cd54f4` | **GEMV kernels for M=1 decode** — The GEMM kernels (16×16 tiles for dense, 64×64 for W4A16) wasted 94–98% of threads when M=1. New GEMV kernels: 4 outputs per block, 64 threads cooperatively reduce K. Plus async copies (eliminated ~150 cuStreamSynchronize calls) and GPU-side MoE top-K (eliminated 96 blocking D2H transfers). | 3.6 | 18.6 | **5.2x** |
| `71774b4` | **Fix W4A16 weight layout** — Kernel expected `[N, K/2]` row-major but weights were stored as `[K, N/2]`. Every weight read was striding across the wrong dimension, destroying cache locality. A one-line layout fix. | 18.6 | 22.1 | +19% |
| `3caed31` | **Batched MoE expert GEMV** — Instead of 10 separate kernel launches per MoE layer (one per expert), a single launch with `blockIdx.y` selecting the expert. Device-side pointer tables let the GPU find each expert's weights without CPU involvement. | 22.1 | 27.2 | +23% |
| `6755630` | **128-bit vectorized loads** — All GEMV kernels upgraded from 32-bit loads to `uint4` (128-bit) loads, reading 8 BF16 values or 8 FP4 values per memory transaction. | 27.2 | 27.9 | +3% |
| `49d46ba` | **Pre-upload attention metadata** — Positions, slot mapping, seq lens, block table uploaded once before the layer loop instead of 4 H2D copies × 12 attention layers = 48 copies. | — | — | infra |
| `27721df` | **Fuse RMS norm + residual save** — Combined two kernels into one, eliminating 96 kernel launches and one full read+write of the hidden state per layer. | 27.9 | 28.1 | +1% |
| `73c6c33` | **CUDA graph capture/replay** — The entire decode step (all 48 layers + final norm + LM head + argmax) captured as a single CUDA graph. Eliminates ~700 kernel launches at ~3μs each. Graph keyed by `max_blocks_per_seq` for the single sequence. | 28.1 | 30.6 | +9% |
| `ba1e727` | **FP8 weight GEMV** — New kernel for dense BF16 weights: reads them as FP8 E4M3 (1 byte vs 2 bytes), halving weight bandwidth. Per-row FP8 scales computed at load time. | 30.6 | 34.9 | +14% |
| `ab91b04` | **FP8-quantize ALL dense weights** — Extended FP8 quantization from just QKVZ to every BF16 dense weight in the model (Q, K, V projections in attention, gate projection in MoE, LM head). | 34.9 | 36.9 | +6% |
| `6aa10af` | **Parallel top-K kernel** — Rewrote MoE top-K routing from single-thread scan (50μs) to parallel warp-shuffle reduction with cross-warp shared memory merge (9μs). | 36.9 | 40.0 | +8% |
| `05be359` | **BLOCK_SIZE 128 for MoE GEMV** — Reduced from 256 to 128 threads per block. Each output element uses exactly 1 warp (32 threads) instead of 2. Eliminates `__syncthreads()` and shared memory for cross-warp reduction. Doubles concurrent CTAs per SM. | 40.0 | 41.0 | +3% |

**Stage 1 total: 11.4x speedup** (3.6 → 41.0 tok/s).

At this point, Atlas Spark at 41 tok/s already exceeded vLLM's 36.4 tok/s non-speculative baseline.

### Stage 2: The E2M1 LUT Breakthrough (41 → 80 tok/s, 1.95x)

| Commit | What Changed | Before | After | Gain |
|--------|-------------|--------|-------|------|
| `260f49e` | **Shared memory E2M1 LUT** — This was the single biggest optimization. NVFP4 dequantization requires converting 4-bit E2M1 values to float via a 16-entry lookup table. The LUT was in `__device__ __constant__` memory, which has limited bandwidth. Moving it to `__shared__` memory (loaded once per block, 64 bytes) gave **+71%** throughput. The constant memory bottleneck was invisible in profiling — it showed up as "compute" time because constant cache misses stall the entire warp. | 41.0 | 70.3 | **+71%** |
| `e24836d` | **NVFP4-quantize dense weights** — BF16 dense weights (previously FP8-quantized) further quantized to NVFP4 at load time. Uses the W4A16 GEMV kernel path instead of the FP8 path. Halves weight bandwidth again (0.5 bytes vs 1 byte per element). Requires computing FP8 per-group scales and per-tensor scale2 at quantization time. | 70.3 | 80.0 | +14% |

**Stage 2 total: 1.95x** (41 → 80 tok/s). The E2M1 LUT optimization alone was worth more than all Stage 1 optimizations combined.

### Stage 3: Kernel Fusion Campaign (80 → 99.1 tok/s, 1.24x)

With individual kernels near their bandwidth limits, the remaining gains came from eliminating kernel boundaries — fusing multiple operations into single launches to reduce graph nodes and intermediate memory traffic.

| Commit | What Changed | Before | After | Gain |
|--------|-------------|--------|-------|------|
| `7354818` | **Fuse MoE expert gate+up and silu+down** — Two separate batched GEMV launches (gate, up) merged into one with `blockIdx.z` selecting the projection. Similarly for silu+down. Halves MoE kernel launches per layer. | 80.0 | 85.0 | +6% |
| `c21c515` | **Fuse shared expert into routed expert kernels** — The shared expert (one per MoE layer) runs as an extra `blockIdx.y` slot alongside the 10 routed experts. Eliminates 2 separate kernel launches per layer × 48 layers = 96 graph nodes. | 85.0 | 86.4 | +2% |
| `f64c294` | **Transpose GDN state for coalesced memory** — SSM recurrent state was stored in a layout that caused strided memory access during the GDN update. Transposing to `[num_heads, head_dim_k, head_dim_v]` gave coalesced reads/writes. | 86.4 | 87.5 | +1% |
| `66ef3f1` | **Fuse BA projection + GDN gates, QKVZ + deinterleave** — BA projection GEMV and gate/beta computation (sigmoid, softplus, exp) fused into a single kernel. QKVZ projection writes directly to deinterleaved output layout, eliminating the separate deinterleave kernel. | 87.5 | 88.9 | +2% |
| `433d12a` | **Eliminate graph re-captures for batch=1** — For single-sequence inference, `max_blocks_per_seq` only appears as `seq_idx * max_blocks` which is always zero. The captured graph works for any block table size, eliminating recapture overhead. | 88.9 | 89.9 | +1% |
| `a115834` | **Fuse wsum+blend and K+V dual GEMV** — MoE weighted sum, sigmoid blend, and shared gate scalar all fused into a single kernel. K and V projections in attention fused into one launch with `blockIdx.z`. | 88.9 | 94.8 | +7% |
| `982af16` | **Fuse gate scalar GEMV into wsum+blend** — The shared expert gate (a 1×1 GEMV producing a single scalar) computed redundantly by every block in the wsum+blend kernel, avoiding a separate launch entirely. | 94.8 | 95.1 | +0.3% |
| `71694c0` | **Fast math intrinsics** — Replaced `expf()`, `logf()` with `__expf()`, `__logf()` (single-precision fast intrinsics) in all hot-path kernels. ~1 ULP less precise but measurably faster. | — | — | small |
| `53ca97e` + `a0e6d30` | **Register-tiled 2-row GEMV** — Each thread group processes 2 output rows instead of 1, sharing the same activation reads and scale computation. Halves the effective block count and improves register-level ILP. | — | — | cumulative |
| `2b339a4` | **Fuse shared expert into routed expert kernels (final)** — The culmination of the fusion campaign. Routed and shared experts run in a single kernel with the shared expert as `blockIdx.y == top_k`. | 96.0 | **99.1** | +3% |
| `38ec848` | **SiLU precompute in shared memory** — For silu+down kernels, precompute `SiLU(gate) * up` once cooperatively into shared memory instead of each thread group computing it independently from global memory. | — | — | cumulative |
| `6fbc097` | **Shared memory A preload** — All GEMV kernels preload the shared input activation vector into shared memory as pre-converted float32. All 4 thread groups in a block read from shared memory instead of L2, eliminating 3/4 redundant loads and BF16→float conversions. | — | ~96.6 | cumulative |

**Stage 3 total: 1.24x** (80 → 99.1 tok/s peak, ~96–97 sustained).

---

## The Final Numbers

| Metric | Value |
|--------|-------|
| **Peak throughput** | **99.1 tok/s** |
| **Sustained throughput** | **~96–97 tok/s** |
| **Replay latency (p50)** | ~10.3 ms |
| **CUDA graph nodes** | ~700 per decode step |
| **Active weight reads per token** | ~4 GB |
| **Effective bandwidth utilization** | ~75–80% of 273 GB/s peak |
| **Total speedup vs initial** | **27.5x** (3.6 → 99.1) |
| **Unit tests** | 43 passing |
| **CUDA kernels** | 20 production kernels |
| **PTX modules** | 29 registered |
| **Lines of CUDA** | ~4,000 (production kernels) |
| **Lines of Rust** | ~10,000+ |

### Comparison with existing frameworks

| Framework | Quantization | Throughput | Speculative Decoding |
|-----------|-------------|-----------|---------------------|
| **Atlas Spark** | **NVFP4** | **99.1 tok/s** | **No** |
| vLLM v22 + Marlin + MTP | NVFP4 | 59.9 tok/s | Yes (2 draft tokens) |
| vLLM v22 + Marlin | NVFP4 | 36.4 tok/s | No |
| TensorRT-LLM v1.3.0rc2 | NVFP4 | 29.6 tok/s | No |
| Atlas vLLM backend | NVFP4 | 40.5 tok/s | No |

Atlas Spark at 99 tok/s **without speculative decoding** is 65% faster than vLLM's best result **with** speculative decoding (59.9 tok/s with MTP).

---

## What Made the Difference

### 1. Owning the full stack

No framework overhead. No Python. No torch.compile boundaries. No NCCL. Every memory copy, every kernel launch, every synchronization point is explicit and intentional.

### 2. Understanding the hardware

LPDDR5X at 273 GB/s is the hard constraint. Every optimization was evaluated against the bandwidth model: *how many bytes does this kernel read, and how close is it to the theoretical minimum latency?* This prevented chasing dead ends.

### 3. The E2M1 LUT insight

The +71% speedup from moving a 64-byte lookup table from constant memory to shared memory was the project's most surprising result. Constant memory has a dedicated cache that's fast for uniform access (all threads reading the same address) but becomes a serialization bottleneck when threads access different entries — exactly what happens during NVFP4 dequantization, where each thread looks up a different 4-bit weight value.

### 4. Aggressive kernel fusion

By the end of the optimization campaign, many of the original ~1,400 kernel launches per decode step were fused down to ~700. Each fusion eliminated not just launch overhead, but intermediate memory traffic — data that never needed to touch global memory because it could stay in registers or shared memory.

### 5. NVFP4 everywhere

Quantizing ALL weights to NVFP4 (including BF16 dense weights that were originally left at higher precision) was counterintuitive — it adds dequantization compute. But halving the bytes read from memory more than compensates, because the GPU is memory-bandwidth-bound, not compute-bound.

---

## What About Speculative Decoding?

**No — speculative decoding has NOT been implemented in Atlas Spark.**

It was extensively explored in TensorRT-LLM (19 experiments with NGram self-speculation, versions v3 through v21), but those attempts all failed:

- **NGram matching** achieves only ~27% acceptance rate on this model — the output is too unpredictable for n-gram patterns
- With 73% rejection rate, the overhead of verifying draft tokens + rolling back SSM state across 36 Mamba layers creates a **net slowdown** (20.2 tok/s vs 29.5 tok/s baseline)
- The hybrid architecture (attention + Mamba SSM) makes speculation uniquely difficult because SSM state is recurrent and must be checkpointed/rolled back on rejection

### What would work

The recommended approach for Atlas Spark is **EAGLE-2 or Medusa** with trained draft heads (not n-gram matching):

- A small MLP draft head on top of the last layer's hidden states predicts 2–4 future tokens
- With a properly trained head, 70–80% acceptance rate is achievable
- SSM state checkpointing is straightforward in Atlas (just memcpy ~2.3 MB of state buffers before each speculative batch)
- **Projected throughput with 3 draft tokens at 80% acceptance: ~160–200 tok/s**

This is the clear next step for Atlas Spark, and it's the one optimization that could deliver another 2x+ improvement.

---

## The Complete Commit History

Every step, from the first kernel benchmark to the final shared memory optimization:

```
Phase 1: Atlas Kernel Library
  98a2913  Atlas v0.4.0-sm121: 18/31 benchmark wins, 51/51 tests passing
  4373d8e  Step 3: Rewrite prefill attention with tensor core MMA
  9e95cbf  Prefill attention: 4-warp PV + fix bench timing
  f654a34  Phase 0+1: Remove sync overhead + add GDR prefill baseline (18→22/32 wins)
  7f3872d  Prefill attention v2: cp.async + register softmax (22→24/32 wins)
  32477a1  Decode attention v2: batched KV + tree merge (24→25/32 wins)

Phase 2: vLLM Integration
  098c74b  Full vLLM integration: Atlas attention backend with CUDA graph support
  25f4b8b  Zero-copy attention + FP8 kernel infrastructure + CUDA graph fix
  9a3df32  Fix FP8 KV cache stride mismatch + CUDA graph FULL mode
  bd88b9b  Atlas attention only: 41 tok/s

Phase 3: Atlas Spark Infrastructure
  b1aa889  Atlas Spark: pure Rust inference server for Qwen3-Next-80B

Phase 4: Performance Optimization (3.6 → 99.1 tok/s)
  0cd54f4  GEMV kernels + async copies + GPU MoE top-K             → 18.6 tok/s  (5.2x)
  71774b4  Fix W4A16 kernel layout: [K,N/2] → [N,K/2]             → 22.1 tok/s  (+19%)
  3caed31  Batched MoE expert GEMV                                  → 27.2 tok/s  (+23%)
  6755630  Vectorize GEMV kernels: 128-bit loads                    → 27.9 tok/s  (+3%)
  49d46ba  Pre-upload attention metadata: 48 → 4 H2D copies
  27721df  Fuse RMS norm + residual save: 96 fewer launches         → 28.1 tok/s  (+1%)
  73c6c33  CUDA graph capture/replay                                → 30.6 tok/s  (+9%)
  68b0d76  Fix argmax_on_device to use backend stream
  3b81d83  Fix MoE topK crash: support 512 experts
  ba1e727  FP8 weight GEMV for dense layers                         → 34.9 tok/s  (+14%)
  ab91b04  FP8-quantize ALL BF16 dense weights                     → 36.9 tok/s  (+6%)
  6aa10af  Parallel top-K kernel (warp shuffle reduction)           → 40.0 tok/s  (+8%)
  05be359  MoE expert GEMV: BLOCK_SIZE 256→128, warp-only reduce   → 41.0 tok/s  (+3%)
  260f49e  Shared memory E2M1 LUT                                   → 70.3 tok/s  (+71%)  ★
  e24836d  NVFP4-quantize BF16 dense weights                       → 80.0 tok/s  (+14%)
  7354818  Fuse MoE expert gate+up and silu+down                    → 85.0 tok/s  (+6%)
  c21c515  Fuse shared expert gate+up and silu+down                 → 86.4 tok/s  (+2%)
  f64c294  Transpose GDN state for coalesced memory                 → 87.5 tok/s  (+1%)
  66ef3f1  Fuse BA+gates and QKVZ+deinterleave                     → 88.9 tok/s  (+2%)
  433d12a  Eliminate graph re-captures for batch=1                   → 89.9 tok/s  (+1%)
  a115834  Fuse wsum+blend and K+V dual GEMV                        → 94.8 tok/s  (+7%)
  982af16  Fuse gate scalar GEMV into wsum+blend                    → 95.1 tok/s
  71694c0  Use __expf/__logf fast intrinsics
  53ca97e  Register-tiled gate+up expert GEMV: 2 rows per thread
  a0e6d30  Register-tiled SiLU+down expert GEMV: 2 rows per thread
  2b339a4  Fuse shared expert into routed expert kernels             → 99.1 tok/s  (+3%)  ★
  38ec848  Precompute SiLU(gate)*up in shared memory
  6fbc097  Preload activation A into shared memory across all GEMV   → ~96.6 tok/s
```

### The top 5 optimizations by impact

| Rank | Optimization | Gain | Why it worked |
|------|-------------|------|---------------|
| 1 | **Shared memory E2M1 LUT** | **+71%** | Constant memory serialization bottleneck invisible in profiling |
| 2 | **GEMV kernels for M=1** | **5.2x** | GEMM tiles wasted 94–98% of threads at batch=1 |
| 3 | **Batched MoE expert GEMV** | **+23%** | 10 launches → 1 launch per projection |
| 4 | **Fix W4A16 weight layout** | **+19%** | Wrong memory access pattern destroyed cache locality |
| 5 | **NVFP4-quantize dense weights** | **+14%** | Halved weight bytes at the cost of dequant compute (worth it) |

---

## Build & Run

```bash
# Start build container
sudo docker run -d --name atlas-build --gpus all --ipc=host --entrypoint bash \
  -v /workspace/atlas:/workspace/atlas avarok/dgx-vllm-nvfp4-kernel:v22 -c "sleep 86400"
sudo docker exec atlas-build bash -c \
  "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"

# Build (clear PTX cache when CUDA sources change)
sudo docker exec -w /workspace/atlas atlas-build bash -c \
  "source /root/.cargo/env && rm -rf target/release/build/atlas-kernels-* && cargo build --release"

# Unit tests (43 tests, no GPU required)
sudo docker exec -w /workspace/atlas atlas-build bash -c \
  "source /root/.cargo/env && cargo test -p spark-model -p spark-runtime -p atlas-kernels \
   -p spark-server -p spark-comm --release"

# Integration benchmark (GPU + weights, 200 tokens)
sudo docker exec -w /workspace/atlas atlas-build bash -c \
  "source /root/.cargo/env && RUST_LOG=info cargo test -p spark-server --release -- --ignored"
```

---

*Built with Rust, CUDA, and an obsession with memory bandwidth.*
*February 2026*
