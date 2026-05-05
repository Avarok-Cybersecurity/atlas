# Atlas Kernel Registry

Tracking every kernel, its baseline comparison, and optimizations applied.

**Target**: nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 on DGX Spark GB10 (SM121)
**Goal**: Beat vLLM v21+Marlin+MTP (59.9 tok/s) with purpose-built kernels
**Version**: 0.3.0-sm121-clean (8 CUDA kernels, best-only inventory)

## Hardware: GB10 (SM121)

| Attribute | Value |
|-----------|-------|
| SMs | 48 |
| Shared Memory | 99 KB/SM |
| Memory BW | 273 GB/s LPDDR5X |
| Tensor Cores | mma.sync m16n8k16 (BF16), m16n8k32 (E2M1 FP4) |
| Registers | 255/thread |
| Clusters | 1x1x1 only |
| TMEM | None (register-to-register mma.sync only) |

## Kernel Inventory (8 kernels)

### 1. dense_gemm_tc_bf16 — BF16 Tensor Core GEMM
- **File**: `cuda_kernels/dense_gemm_tc.cu`
- **Instruction**: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
- **Status**: CORRECT, all sizes pass (including non-aligned)
- **CTA Tile**: 64×64, K_STEP=64, 128 threads (4 warps)
- **Features**: Double-buffered shared memory, 32-bit vectorized loads with alignment-safe fallback
- **Shared Memory**: 2×(64×66 + 64×66) × 2B = 33 KB
- **Accumulation**: FP32 → BF16 output
- **Correctness**: Bit-exact at 64×64, 0.27% rel error at 80×2048×512 (BF16 rounding)
- **Key Fix**: Fragment register ordering a[0]=rowG/Klo, a[1]=rowG+8/Klo, a[2]=rowG/Khi, a[3]=rowG+8/Khi

#### Benchmarks

| Shape (M×N×K) | Atlas TC | cuBLAS | Notes |
|---------------|----------|--------|-------|
| 64×64×64 | 0.011ms | 0.009ms | 1.2× cuBLAS |
| 80×512×2048 | 0.120ms | 0.009ms | MoE gate_up projection |
| 80×2048×512 | 0.034ms | 0.006ms | MoE down projection |
| 256×256×256 | 0.024ms | 0.006ms | Medium GEMM |

### 2. dense_gemm_bf16 — Scalar BF16 GEMM (fallback)
- **File**: `cuda_kernels/dense_gemm_bf16.cu`
- **Tile**: 16×16×16, one thread per output element
- **Status**: CORRECT (bit-exact), used as fallback when K < 16
- **Note**: Auto-dispatch sends K≥16 to TC path

### 3. rms_norm — RMS Normalization
- **File**: `cuda_kernels/rms_norm.cu`
- **Status**: CORRECT, 0.47% rel error (BF16 rounding)
- **Shape**: [num_tokens, hidden_size], one block per token

### 4. fused_silu_mul — Fused SiLU(gate) × up
- **File**: `cuda_kernels/dense_gemm_bf16.cu` (shares file with scalar GEMM)
- **Status**: CORRECT, 0.56% rel error
- **Shape**: [N, inter_size×2] → [N, inter_size]

### 5. e2m1_branchless — FP32 → E2M1 (FP4) Quantization
- **File**: `cuda_kernels/e2m1_branchless.cu`
- **Status**: CORRECT, bit-exact
- **Note**: 7 uint comparisons, no branches. Packs 8 values into 1 uint32.

### 6. moe_permute_tokens — Token Expert Routing
- **File**: `cuda_kernels/moe_permute.cu`
- **Status**: CORRECT, bit-exact
- **Kernels**: `moe_permute_tokens`, `moe_unpermute_reduce`, `moe_count_experts`

### 7. w4a16_gemm — Fused W4A16 Dequant+GEMM
- **File**: `cuda_kernels/w4a16_gemm.cu`
- **Instruction**: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (BF16 TC)
- **Status**: CORRECT, bit-exact at 16×16×16, 0.1% mean error at MoE scale
- **CTA Tile**: 64×64, K_STEP=16, 128 threads (4 warps)
- **Strategy**: Load packed NVFP4 (E2M1) weights → dequant to BF16 in smem → BF16 MMA
- **Dequant**: E2M1_LUT[nibble] × FP8_scale × scale2 → BF16
- **Purpose**: Single-expert W4A16 GEMM (building block for grouped GEMM)
- **Key insight**: Native FP4 MMA (`kind::f8f6f4`) NOT available on SM121. Must use W4A16 dequant path.

### 8. moe_w4a16_grouped_gemm — Grouped W4A16 GEMM for MoE
- **File**: `cuda_kernels/moe_w4a16_grouped_gemm.cu`
- **Status**: CORRECT — bit-exact at small sizes, <0.1% mean error at 256 experts
- **Grid**: (ceil(N/64), max_m_tiles, num_experts) — 3D grid, one CTA tile per expert
- **expert_offsets**: [num_experts+1] prefix sum maps blockIdx.z to row range in A/C
- **Weight indexing**: B_packed + expert_id × weight_stride per expert
- **Single kernel launch** handles ALL 256 experts simultaneously
- **Winner** of 5 MoE GEMM approaches tested (see comparison below)

#### Grouped GEMM Benchmarks (Qwen3-Next shapes)

| Operation | Atlas W4A16 | cuBLAS per-expert (BF16) | Speedup |
|-----------|------------|--------------------------|---------|
| Gate-up: 800×1024×2048 | **5.58ms** | 7.14ms | **1.28×** |
| Down: 800×2048×512 | **2.83ms** | — | — |
| Full pipeline | **8.39ms** | — | — |

**Atlas beats cuBLAS** for MoE workload because:
1. Single kernel launch vs 256 per-expert cuBLAS launches
2. 3.6× less weight data to read (FP4 vs BF16)
3. Fused dequant in shared memory (no intermediate BF16 materialization)

#### Memory Analysis (256 experts)

| | W4A16 (FP4+FP8 scales) | BF16 | Reduction |
|---|---|---|---|
| Gate-up weights | 302 MB | 1074 MB | 3.6× |
| Down weights | 151 MB | 537 MB | 3.6× |
| **Total** | **453 MB** | **1611 MB** | **3.6×** |

## Rust Pipeline: moe_forward_w4a16

Chains 5 kernels with single `stream.synchronize()`:
```
permute → grouped_W4A16_GEMM(gate_up) → SiLU → grouped_W4A16_GEMM(down) → unpermute
```
- **Status**: CORRECT — end-to-end pipeline verified (bit-exact at small sizes)
- **Single Python call**: Eliminates Python overhead, no per-kernel sync
- **Total time** (Qwen3-Next shapes): 8.39ms for full MoE layer

## Correctness Verification

| Pipeline | GEMM Backend | Max Rel Error | Status |
|----------|-------------|---------------|--------|
| Scalar | dense_gemm_bf16 (16×16 tiled) | 0.0000% (bit-exact) | PASS |
| Tensor Core | dense_gemm_tc_bf16 (mma.sync) | 0.0000% (bit-exact) | PASS |
| TC at MoE scale | 80×512×2048 → SiLU → 80×2048×512 | 0.2618% | PASS |
| **W4A16 grouped** | moe_w4a16_grouped_gemm | **0.0000%** (small), **0.10%** (256 experts) | **PASS** |
| **W4A16 full pipeline** | moe_forward_w4a16 (5 kernels) | **0.0000%** (bit-exact, small) | **PASS** |

## MoE GEMM Approach Comparison

All approaches tested for gate_up projection: 800×1024×2048 across 256 experts.
Only the winner (3D grid) is kept in the codebase; others were deleted after benchmarking.

| Approach | Time | vs Winner | Notes |
|----------|------|-----------|-------|
| **3D grid (WINNER)** | **5.56ms** | **1.00×** | Grid (16, 1, 256) = 4096 CTAs |
| K_STEP=64 + double-buffer | 7.09ms | 0.79× | Dequant overhead reduces occupancy |
| Persistent: 48 CTAs + atomic queue | 14.42ms | 0.39× | Atomic contention + serial per-CTA |
| Serial experts: Rust loop, no sync | 45.18ms | 0.12× | 245 launch dispatches dominate |
| cuBLAS per-expert (BF16) | 7.14ms | 0.78× | Python→cuBLAS × 245 experts |

**Key finding**: The GPU's hardware CTA scheduler is the most efficient work distributor.
A single 3D grid launch with early-exit for empty experts beats all alternatives.

**Bandwidth analysis**: Achieved 54 GB/s (20% of 273 GB/s peak).
Theoretical minimum: 1.11ms (reading 302 MB at peak).
Room for 5× improvement via better memory access patterns and reduced dequant overhead.

## Architecture Decisions

1. **Rust orchestration, CUDA C kernels only**: All dispatch, shape logic, error handling in Rust. .cu files contain only `__global__`/`__device__` functions.

2. **Raw u64 device pointers across FFI**: Zero-copy tensor passing from PyTorch. No Python object inspection in the hot path.

3. **PTX compilation at build time**: nvcc --ptx -arch=compute_120. Embedded via include_str! in Rust. Single `cargo build --release` compiles everything.

4. **Auto-dispatch**: `dense_gemm_bf16()` checks K≥16 → tensor cores, else scalar fallback.

5. **Global AtlasRegistry (OnceLock)**: All PTX modules, CUDA context, and stream cached in a singleton. First call loads everything (~168ms), subsequent calls are instant.

6. **Best-only inventory**: Only the fastest version of each kernel is kept. Slower variants are benchmarked, documented, then deleted.

## Optimization Log

| Date | Change | Impact |
|------|--------|--------|
| 2026-02-23 | TC GEMM: mma.sync.m16n8k16 BF16 | Correct tensor core GEMM on SM121 |
| 2026-02-23 | Fragment mapping fix: swap a[1]↔a[2] | Fixed 80-100% error → bit-exact |
| 2026-02-23 | Auto-dispatch: dense_gemm_bf16 → TC when K≥16 | All GEMM callers get TC automatically |
| 2026-02-23 | AtlasRegistry: OnceLock singleton | **2.2-7.8× speedup** across all kernel calls |
| 2026-02-23 | TC GEMM: K_STEP=64 + double-buffer + vectorized loads | **15-19% speedup** over K_STEP=16 |
| 2026-02-23 | Confirmed: FP4 `kind::f8f6f4` NOT available on SM121 | Must use W4A16 dequant path |
| 2026-02-23 | W4A16 fused dequant+GEMM kernel | Correct, bit-exact small, 0.1% MoE scale |
| 2026-02-23 | Grouped W4A16 GEMM (all experts, 1 launch) | **1.28× faster than cuBLAS per-expert** |
| 2026-02-23 | moe_forward_w4a16: full pipeline in Rust | 5 kernels, single sync, 8.39ms end-to-end |
| 2026-02-23 | Inventory cleanup: removed 6 slower kernel variants | 14 → 8 CUDA files, 13 → 8 PTX modules |
| | Vectorized B loads (TODO) | 128-bit coalesced reads for packed FP4 |
| | Fuse SiLU into GEMM epilogue (TODO) | Eliminate intermediate gate_up buffer write |
| | Bandwidth optimization (20%→40%+) | Reduce dequant overhead, better access patterns |
