# Atlas Spark: Coherence Journey

Tracking the coherence impact on each optimization commit from the [Atlas Spark Journey](./ATLAS_SPARK_JOURNEY.md).

**Current status**: 81.4 tok/s sustained, COHERENT. Output: "The capital of France is Paris."

**Performance ceiling**: 81.4 tok/s is the practical ceiling for single-token decode on GB10 LPDDR5X (273 GB/s peak). Total weight reads ~1,517 MB/token. GEMV bandwidth optimization (4 approaches) and MTP speculation (48% acceptance) both failed to improve throughput. See Stage 4 below.

**Method**: Cherry-pick each optimization from `master` onto `coherence-journey`, rebuild, run coherence test (`coherence_test_capital_of_france`), record result. If coherence breaks, fix before proceeding.

---

## Coherence Baseline (commit `631d14c`)

Two critical bugs fixed that were present in ALL original optimization commits:

1. **Attention Q/Gate interleaving** — HF q_proj outputs per-head interleaved `[Q_h0(256), G_h0(256), ...]` but Atlas assumed `[All_Q, All_Gate]`. Corrupted all 12 attention layers. Fix: `deinterleave_qg` kernel in `ssm_preprocess.cu`.

2. **MoE softmax domain** — Softmax computed over top-K (10) only instead of all 512 experts. Fix: full parallel softmax in `moe_topk.cu`.

Plus accumulated fixes from debugging: GDN decay-before-correction, gated RMS norm SiLU, rotate_half RoPE, sigmoid gate mul kernel.

---

## Stage 1: Eliminating Gross Inefficiencies (3.6 → 41 tok/s)

| # | Original Commit | New Commit | What Originally Changed | Original | What Changed (Coherence) | After Coherence | Delta |
|---|-----------------|------------|------------------------|----------|--------------------------|-----------------|-------|
| 1 | `0cd54f4` | `def40f7` | **GEMV kernels for M=1 decode** — New GEMV kernels (4 outputs/block, 64 threads), async copies, GPU-side MoE top-K. | 3.6 → 18.6 (5.2x) | 2 bugs fixed: MoE buffer overflow + top-K 256→512. Output: garbage. | 19.5 tok/s | — |
| 2 | `71774b4` | `1ab9d37` | **Fix W4A16 weight layout** — `[K,N/2]` → `[N,K/2]`. | 18.6 → 22.1 (+19%) | No new fixes. Output: "HGGGGG...!!!..." | 24.0 tok/s | — |
| 3 | `3caed31` | `340a791` | **Batched MoE expert GEMV** — Single launch with blockIdx.y expert selection. | 22.1 → 27.2 (+23%) | No new fixes. Output: "reimimim..." | — | — |
| 4 | `6755630` | `cfffa81` | **128-bit vectorized loads** — uint4 loads for 8 BF16/FP4 values per transaction. | 27.2 → 27.9 (+3%) | No new fixes. Output: "reimimim..." | 29.3 tok/s | — |
| 5 | `49d46ba` | `9b70ecf` | **Pre-upload attention metadata** — 48 → 4 H2D copies per step. | infra | Bundled with items 6-7 in single cherry-pick. | — | — |
| 6 | `27721df` | `9b70ecf` | **Fuse RMS norm + residual save** — 96 fewer kernel launches. | 27.9 → 28.1 (+1%) | Bundled with items 5,7. | — | — |
| 7 | `73c6c33` | `9b70ecf` | **CUDA graph capture/replay** — Entire decode as single graph. | 28.1 → 30.6 (+9%) | Bundled with items 5-6. | — | — |
| — | — | `631d14c` | **COHERENCE FIX** — Q/Gate deinterleave + full MoE softmax. | — | All above + coherence bugs fixed. First coherent output. | **30.4 tok/s** | **COHERENT** |
| 8 | `ba1e727` | `c399991` | **FP8 weight GEMV** — Dense weights read as FP8 (1 byte vs 2). | 30.6 → 34.9 (+14%) | Clean cherry-pick, no conflicts. | **35.9 tok/s** | **COHERENT** |
| 9 | `ab91b04` | `40f199e` | **FP8-quantize ALL dense weights** — Extended to Q/K/V/gate/lm_head. | 34.9 → 36.9 (+6%) | Conflict in `qwen3_attention.rs`: merged FP8 conditional with Q+Gate deinterleave, fixed q_proj dim from 4096→8192. | **33.5 tok/s** | **COHERENT** |
| 10 | `6aa10af` | `dc1e892` | **Parallel top-K kernel** — Warp-shuffle + cross-warp reduction. | 36.9 → 40.0 (+8%) | 3 conflicts: `moe_topk.cu` (kept our full softmax), `moe.rs`, `model.rs` (merged profiling). | **33.6 tok/s** | **COHERENT** |
| 11 | `05be359` | `928f7a8` | **BLOCK_SIZE 128 for MoE GEMV** — Warp-only reduce, 2x CTAs/SM. | 40.0 → 41.0 (+3%) | Clean cherry-pick. | **36.5 tok/s** | **COHERENT** |

**Stage 1 result: 36.5 tok/s coherent** (vs 41.0 incoherent original — 89% retained)

---

## Stage 2: The E2M1 LUT Breakthrough (41 → 80 tok/s)

| # | Original Commit | New Commit | What Originally Changed | Original | What Changed (Coherence) | After Coherence | Delta |
|---|-----------------|------------|------------------------|----------|--------------------------|-----------------|-------|
| 12 | `260f49e` | `152f4de` | **Shared memory E2M1 LUT** — NVFP4 dequant LUT from constant to shared memory. | 41.0 → 70.3 (+71%) | Clean cherry-pick. | **52.3 tok/s** | **COHERENT** |
| 13 | `e24836d` | `369970d` | **NVFP4-quantize dense weights** — BF16 → NVFP4 at load time, W4A16 kernel. | 70.3 → 80.0 (+14%) | 3 conflicts: `qwen3_attention.rs` (kept Q+Gate ×2 dim + NVFP4), `qwen3_ssm.rs`, `weight_loader.rs` (q_proj 8192 rows). | **74.0 tok/s** | **COHERENT** |

**Stage 2 result: 74.0 tok/s coherent** (vs 80.0 incoherent original — 93% retained)

---

## Stage 3: Kernel Fusion Campaign (80 → 99.1 tok/s)

| # | Original Commit | New Commit | What Originally Changed | Original | What Changed (Coherence) | After Coherence | Delta |
|---|-----------------|------------|------------------------|----------|--------------------------|-----------------|-------|
| 14 | `7354818` | `31087bc` | **Fuse MoE expert gate+up and silu+down** — blockIdx.z projection select. | 80.0 → 85.0 (+6%) | Clean cherry-pick. | **67.2 tok/s** | **COHERENT** |
| 15 | `c21c515` | `5842760` | **Fuse shared expert** — Shared expert as extra blockIdx.y slot. | 85.0 → 86.4 (+2%) | Conflict: `ssm_ba` too small for gate scratch. Used `logits()` buffer. | **77.6 tok/s** | **COHERENT** |
| 16 | `f64c294` | `45de843` | **Transpose GDN state** — Coalesced memory access for SSM state. | 86.4 → 87.5 (+1%) | 2 conflicts: kept `g_t * hk_dot` gated residual in prefill kernel, kept `l2_norm_k` handle. | **75.5 tok/s** | **COHERENT** |
| 17 | `66ef3f1` | `337fe4e` | **Fuse BA+gates, QKVZ+deinterleave** — Direct deinterleaved write. | 87.5 → 88.9 (+2%) | 3 conflicts: kept `deinterleave_qg` kernel, took fused BA+gates + QKVZ, added `prof!` macro. | **74.8 tok/s** | **COHERENT** |
| 18 | `433d12a` | `0a6c95f` | **Eliminate graph re-captures** — Single graph for batch=1. | 88.9 → 89.9 (+1%) | Conflict in model.rs: kept debug_layers guard, simplified max_blocks. | **77.1 tok/s** | **COHERENT** |
| 19 | `a115834` | `cfdc2c5` | **Fuse wsum+blend, K+V dual GEMV** — Major fusion pass. | 89.9 → 94.8 (+7%) | 2 conflicts: kept `qg_bytes` K offset, kept `logits()` for gate scratch. | **77.3 tok/s** | **COHERENT** |
| 20 | `03d1bcb` | `816268e` | **Fuse residual+norm** — Single kernel for residual add + norm. | small | **ROOT CAUSE FIXED**: original kernel used `w` instead of `(1+w)` — missing Qwen3-Next offset-from-1 RMS norm. Fixed and re-applied. | **79.7 tok/s** | **COHERENT** |
| 21 | `982af16` | `e93b7fb` | **Fuse gate scalar into wsum+blend** — Inline 1×1 GEMV. | 94.8 → 95.1 (+0.3%) | Clean cherry-pick. | **79.0 tok/s** | **COHERENT** |
| 22 | `71694c0` | `3c1882b` | **Fast math intrinsics** — `__expf()`, `__logf()` in hot paths. | small | Conflict in moe_topk.cu: kept full softmax, applied `__expf`. | **78.6 tok/s** | **COHERENT** |
| 23 | `53ca97e` | `ee7a08b` | **Register-tiled gate+up GEMV** — 2 output rows per thread group. | cumulative | Clean cherry-pick. | — | — |
| 24 | `a0e6d30` | `de1b574` | **Register-tiled silu+down GEMV** — 2 output rows per thread group. | cumulative | Clean cherry-pick. | — | — |
| 25 | `2b339a4` | `e559488` | **Fuse shared expert (final)** — Routed + shared in single kernel. | 96.0 → 99.1 (+3%) | Conflict: fixed shared_gate_scratch to logits() buffer. | **80.0 tok/s** | **COHERENT** |
| 26 | `38ec848` | `9fe87b4` | **SiLU precompute in shared memory** — Cooperative precompute. | cumulative | Clean cherry-pick. | — | — |
| 27 | `6fbc097` | `3239bd7` | **Shared memory A preload** — All GEMVs preload activation to smem. | ~96.6 | Clean cherry-pick. | **78.8 tok/s** | **COHERENT** |
| 20' | `03d1bcb` | `816268e` | **Fuse residual+norm (FIXED)** — Re-applied with `(1+w)` fix. | — | Root cause: kernel used `w` not `(1+w)`. Eliminates 96 graph nodes. | **79.7 tok/s** | **COHERENT** |
| 27' | — | `8342fc8` | **Revert shared memory A preload** — s_A[2048-4096] increased smem 100B→16KB, reduced occupancy. | — | Direct global A reads competitive on LPDDR5X. SiLU precompute (#26) retained. | **80.8 tok/s** | **COHERENT** |
| 28 | — | `45b3d4d` | **Optimize moe_topk + fuse deinterleave_qg** — Eliminate redundant softmax phases + fused Q/Gate GEMV deinterleave. | — | moe_topk: skip Phase 3a/3b (reuse top-K[0] as max). w4a16_gemv_qg: inline deinterleave saves 12 graph nodes. | **81.6 tok/s** | **COHERENT** |

**Stage 3 result: 81.6 tok/s coherent** (vs 99.1 incoherent original — 82% retained)

---

## Risk Assessment

Commits likely to conflict with coherence fixes:
- **`6aa10af`** (parallel top-K) — `moe_topk.cu` was rewritten for full softmax; may conflict
- **`66ef3f1`** (QKVZ+deinterleave fusion) — We added `deinterleave_qg` to `ssm_preprocess.cu`; the QKVZ deinterleave fusion may need adaptation to preserve QG deinterleave
- **`f64c294`** (GDN state transpose) — Our GDN decay fix modified `gated_delta_rule.cu`
- **`03d1bcb`** (fuse residual+norm) — `rms_norm.cu` modified for gated norm + (1+w) style

All others should cherry-pick cleanly or with trivial conflict resolution.

---

## Stage 4: GEMV Bandwidth Optimization (CONCLUDED)

**Goal**: Push past 81.6 tok/s by improving GEMV bandwidth utilization (45-68% of peak 273 GB/s).

**Profiling breakdown** (12.2ms/token):
| Operation | Time/layer | Layers | Total | % decode | Bandwidth% |
|-----------|-----------|--------|-------|----------|------------|
| MoE gate+up | 74μs | 48 | 3.6ms | 29% | 57% |
| QKVZ projection | 75μs | 36 | 2.7ms | 22% | 68% |
| MoE silu+down | 44μs | 48 | 2.1ms | 17% | 48% |
| LM head | — | 1 | 0.7ms | 6% | ~81% |
| SSM core | — | 36 | 0.6ms | 5% | — |
| Other | — | — | 2.5ms | 21% | — |

### Approaches tested (ALL FAILED):

| # | Approach | Result | Root Cause |
|---|----------|--------|------------|
| 1 | Wide SiLU+Down (8 threads/output) | 80.2 tok/s (-2%) | Broke memory coalescing: 32B scattered vs 128B coalesced |
| 2 | 4x register tiling (gate+up, silu+down) | 80.9 tok/s (-1%) | CUDA graph already saturates memory controller across blocks |
| 3 | 1-warp main GEMV | Not attempted | Learnings from 1-2 showed individual kernel ILP is not the bottleneck |
| 4 | `#pragma unroll 2` outer K-loops | 79.5 tok/s (-3%) | Doubled register pressure, reduced occupancy from ~12 to ~5-6 blocks/SM |

### Theoretical analysis:
- Total memory traffic per token: ~1,517 MB (weights + GDN state)
- At 273 GB/s peak: **5.6ms theoretical minimum**
- At 70% practical bandwidth: **8.0ms minimum**
- Non-GEMV overhead: ~3.6ms (norms, attention, SSM, graph replay, metadata)
- Practical floor: **~11.5ms** (94 tok/s) — but closing the gap requires hardware-level improvements

### Key insight:
CUDA graph execution provides sufficient memory parallelism across ~819 kernel nodes and 16 SMs. Individual kernel optimizations (wider reads, more ILP, register tiling) don't help because the memory controller is already saturated by concurrent blocks. The 30-55% "unused" bandwidth is lost to LPDDR5X refresh cycles, row buffer conflicts, and protocol overhead — not addressable at the kernel level.

### Minor optimization committed:
- Merged Q+K L2 norm into single kernel launch (-36 graph nodes, ~783 total)
- No measurable throughput impact, but cleaner graph

---

## Stage 5: MTP Speculative Decoding (CONCLUDED — NOT BENEFICIAL)

**Goal**: Use MTP (Multi-Token Prediction) head to speculate 2 tokens ahead, verify, and accept.

**Result**: 48.2 tok/s (41% SLOWER than 81.4 baseline)

| Metric | Value |
|--------|-------|
| Acceptance rate | 48% (30/62 drafts in 31 steps) |
| Tokens per step | 3.0 avg (min 2, max 4) |
| Step cost | ~55ms (4× decode + propose + checkpoint/rollback) |
| Per-token time | 18.3ms (vs 12.2ms without MTP) |

**Root cause**: For GEMV-based single-token decode engines, verification cost per token = regular decode cost. Speculative decoding requires:
- `tokens_per_step / step_cost > 1 / decode_time`
- With 2 drafts, 48% acceptance: `3.0 / 55ms = 54 tok/s < 81.4 tok/s`
- Even with 100% acceptance: `4.0 / 54ms = 74 tok/s < 81.4 tok/s`

Spec decode only helps when verification is BATCHED (GEMM amortizes weight loads). vLLM's 1.64x speedup (36→60 tok/s) works because their FP4 GEMM can batch-verify tokens cheaply. Atlas's per-token GEMV makes each verification token cost the full 12.2ms.
