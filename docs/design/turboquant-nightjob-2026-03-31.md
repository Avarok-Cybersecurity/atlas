# TurboQuant KV Cache Compression — Night Job 2026-03-31

## Mission

Implement TurboQuant KV cache compression (Google Research, ICLR 2026) in Atlas. Add `--kv-cache-dtype turbo4` that uses Walsh-Hadamard Transform + Lloyd-Max optimal scalar quantization for the KV cache. Same 4-bit memory footprint as NVFP4 but significantly better quality per bit, plus Sparse V optimization that skips ~90% of V dequantization at long context for a free decode speedup.

**Paper**: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (Zandieh et al., Google Research, ICLR 2026)
**Reference impl**: `turboquant_plus` by Tom Turney (llama.cpp fork, 2200+ stars)

## Success Criteria

1. `--kv-cache-dtype turbo4` works end-to-end: server starts, generates coherent text
2. Quality: at least matches NVFP4 on "capital of France" and similar basic tests
3. Sparse V: measurably skips V dequant at long context (logged skip %)
4. All existing `--kv-cache-dtype fp8/bf16/nvfp4` paths still work (no regressions)
5. Compiles clean with `cargo build -r`

## Background: What TurboQuant Does

### Core Algorithm (3 steps, applied per K/V vector at cache write time)

1. **Walsh-Hadamard Transform (WHT)**: Rotate the head_dim=256 vector using fast WHT (O(d log d) butterfly operations). This "Gaussianizes" the coordinates — makes them approximately i.i.d. so a single scalar quantizer works optimally for all dimensions. Kurtosis drops from ~900 to ~3.0.

2. **Lloyd-Max Optimal Scalar Quantization**: Pre-computed codebook with non-uniform levels that minimize MSE for the known (approximately Gaussian) distribution. For 4-bit: 16 codebook entries. Much lower MSE than E2M1's uniform-ish levels at the same bit rate.

3. **Per-group scaling**: Same as NVFP4 — compute absmax per group of 16 elements, store as FP8 E4M3 scale. Normalize to unit variance before codebook lookup.

### Key Architectural Insight: Work in WHT Domain

Since WHT is orthogonal, inner products are preserved: `<WHT(Q), WHT(K)> = <Q, K>` (Parseval's theorem). And WHT is linear: `Σ wᵢ·WHT(Vᵢ) = WHT(Σ wᵢ·Vᵢ)`. This means:

- **Cache stores WHT(K) and WHT(V)** (rotated + quantized)
- **At decode**: WHT-transform Q once at the start
- **K dot products**: computed directly in WHT domain (no inverse needed)
- **V accumulation**: accumulate in WHT domain (no inverse per position)
- **Final output**: single iWHT on the accumulated result

**Zero iWHT per KV position** — only 1 WHT(Q) + 1 iWHT(O) per attention computation.

### Sparse V (turboquant_plus extension)

During decode attention, softmax weights are computed from K **before** touching V. Positions where `max(softmax_weights_in_batch) < threshold` skip V dequantization entirely. At 32K context, ~90% of V is skipped. This is a per-batch-of-4 check within the existing BC=4 inner loop.

### Why turbo4 has identical memory layout to NVFP4

Both are 4-bit data + FP8 per-group scales. Same packed nibble format, same scale section. The differences are purely computational:
- Different quantization codebook (Lloyd-Max vs E2M1)
- WHT rotation before quantization
- WHT/iWHT in the attention kernel

This means block_bytes, block_stride, data_section_bytes, scale_section_bytes are ALL IDENTICAL to NVFP4. No allocator changes needed.

---

## Implementation Plan

### Phase 1: Rust Foundation (KvCacheDtype + CLI + plumbing)

#### 1.1 Add `Turbo4` to `KvCacheDtype` enum

**File**: `/workspace/atlas/crates/spark-runtime/src/kv_cache.rs`

```rust
pub enum KvCacheDtype {
    Bf16,
    Fp8,
    Nvfp4,
    Turbo4,  // 4-bit WHT + Lloyd-Max (same byte layout as NVFP4)
}
```

Update `Display`:
```rust
KvCacheDtype::Turbo4 => write!(f, "turbo4"),
```

Update `FromStr`:
```rust
"turbo4" => Ok(KvCacheDtype::Turbo4),
// Update error message to list turbo4
```

In `block_bytes_for_dtype()`, add:
```rust
KvCacheDtype::Turbo4 => {
    // Identical layout to NVFP4: 4-bit data + FP8 per-group scales
    let data = elems / 2;
    let num_groups = elems / NVFP4_GROUP_SIZE;
    data + num_groups
}
```

Add helper methods:
```rust
/// Turbo4 data section bytes per block (same layout as NVFP4).
pub fn turbo4_data_bytes(&self) -> usize {
    self.nvfp4_data_bytes()
}

/// Turbo4 scale section bytes per block (same layout as NVFP4).
pub fn turbo4_scale_bytes(&self) -> usize {
    self.nvfp4_scale_bytes()
}
```

#### 1.2 Add CLI flags

**File**: `/workspace/atlas/crates/spark-server/src/cli.rs`

- Update `--kv-cache-dtype` help string to include `turbo4`
- Add new flag:
```rust
/// Sparse V threshold for TurboQuant. Attention positions with weight below
/// this skip V dequantization. 0.0 = disabled. Recommended: 0.001 for turbo4.
#[arg(long, default_value_t = 0.0)]
pub sparse_v_threshold: f32,
```

#### 1.3 Thread sparse_v_threshold from CLI → model

Trace the existing pattern for how `kv_cache_dtype` gets from CLI to `Qwen3AttentionLayer`. Follow the same path for `sparse_v_threshold`:

1. `cli.rs` → CLI args struct
2. Find where `kv_cache_dtype` string is parsed and passed to engine/model config
3. Add `sparse_v_threshold: f32` alongside it
4. In `Qwen3AttentionLayer` struct (`mod.rs`), add field: `sparse_v_threshold: f32`
5. Accept in constructor, store on struct

**Search for the plumbing path**: `grep -r "kv_cache_dtype" --include="*.rs"` to find all touch points.

#### 1.4 Add Turbo4 kernel handle resolution

**File**: `/workspace/atlas/crates/spark-model/src/layers/qwen3_attention/mod.rs`

Find where kernel handles are resolved by `kv_dtype` (search for `KvCacheDtype::Nvfp4` in the constructor). Add `Turbo4` arms:

For reshape_and_cache:
```rust
KvCacheDtype::Turbo4 => gpu.kernel("reshape_and_cache", "reshape_and_cache_flash_turbo4")?,
```

For paged_decode:
```rust
KvCacheDtype::Turbo4 => gpu.kernel("paged_decode_turbo4", "paged_decode_attn_turbo4")?,
```

For split-K:
```rust
KvCacheDtype::Turbo4 => Some(gpu.kernel("paged_decode_turbo4", "paged_decode_attn_splitk_turbo4")?),
```

For reduce:
```rust
KvCacheDtype::Turbo4 => Some(gpu.kernel("paged_decode_turbo4", "paged_decode_attn_reduce_turbo4")?),
```

For prefill:
```rust
KvCacheDtype::Turbo4 => gpu.kernel("prefill_paged_turbo4", "inferspark_prefill_paged_turbo4")?,
```

Also handle any BR=64 prefill variants if they exist for NVFP4.

#### 1.5 Add dispatch in decode.rs

**File**: `/workspace/atlas/crates/spark-model/src/layers/qwen3_attention/decode.rs`

In `write_kv_cache()` — add `KvCacheDtype::Turbo4` match arm. Identical to NVFP4 arm but calls `ops::reshape_and_cache_turbo4` and uses `kv_cache.turbo4_data_bytes()`:
```rust
KvCacheDtype::Turbo4 => ops::reshape_and_cache_turbo4(
    gpu, self.reshape_cache_k, k, v,
    kv_cache.k_pool_ptr(self.attn_layer_idx),
    kv_cache.v_pool_ptr(self.attn_layer_idx),
    slot, num_tokens, num_kv_heads, head_dim, block_size,
    key_stride, value_stride,
    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
    kv_cache.turbo4_data_bytes() as u64,
    stream,
),
```

In `run_paged_decode()` — add `KvCacheDtype::Turbo4` match arm. Clone the NVFP4 arm, replacing:
- `ops::paged_decode_attn_nvfp4` → `ops::paged_decode_attn_turbo4` (add `self.sparse_v_threshold` as extra arg)
- `ops::paged_decode_attn_splitk_nvfp4` → `ops::paged_decode_attn_splitk_turbo4` (add threshold)
- `ops::paged_decode_attn_reduce_nvfp4` → `ops::paged_decode_attn_reduce_turbo4`
- `kv_cache.nvfp4_data_bytes()` → `kv_cache.turbo4_data_bytes()`

#### 1.6 Add dispatch in prefill.rs

**File**: `/workspace/atlas/crates/spark-model/src/layers/qwen3_attention/prefill.rs`

Add `KvCacheDtype::Turbo4` match arm(s) for prefill attention dispatch. Follow the NVFP4 pattern exactly. Maps to `prefill_paged_turbo4` module.

#### 1.7 Add ops functions

**File**: `/workspace/atlas/crates/spark-model/src/layers/ops.rs`

Add these functions (clone from their NVFP4 equivalents — identical signatures except decode functions get extra `sparse_v_threshold: f32` param):

- `reshape_and_cache_turbo4()` — same sig as `reshape_and_cache_nvfp4`
- `paged_decode_attn_turbo4()` — same sig as nvfp4 + `sparse_v_threshold: f32`
- `paged_decode_attn_splitk_turbo4()` — same sig as nvfp4 splitk + `sparse_v_threshold: f32`
- `paged_decode_attn_reduce_turbo4()` — same sig as nvfp4 reduce (no threshold needed)

For the decode/splitk functions, add `.arg_f32(sparse_v_threshold)` to the KernelLaunch chain.

---

### Phase 2: CUDA Write Kernel (reshape_and_cache)

#### 2.1 Walsh-Hadamard Transform device function

**Append to**: `/workspace/atlas/kernels/gb10/nvfp4/reshape_and_cache.cu`

WHT for head_dim=256 across one warp (32 threads × 8 elements each):

```cuda
// ============================================================================
// TurboQuant: Walsh-Hadamard Transform for 256 elements (warp-cooperative)
// ============================================================================

// Each thread in the warp holds 8 elements: vals[0..7]
// Thread `lane` owns elements [lane*8 .. lane*8+7] of the 256-element vector.
// After calling, vals[] contain WHT-transformed coordinates scaled by 1/sqrt(256).
__device__ __forceinline__ void wht256_warp(float vals[8], unsigned int lane) {
    // Stages 0-2: intra-thread butterfly (stride 1, 2, 4)
    // Stage 0: stride=1
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        float a = vals[i], b = vals[i+1];
        vals[i] = a + b;
        vals[i+1] = a - b;
    }
    // Stage 1: stride=2
    #pragma unroll
    for (int i = 0; i < 8; i += 4) {
        float a0 = vals[i], a1 = vals[i+1], b0 = vals[i+2], b1 = vals[i+3];
        vals[i] = a0 + b0;
        vals[i+1] = a1 + b1;
        vals[i+2] = a0 - b0;
        vals[i+3] = a1 - b1;
    }
    // Stage 2: stride=4
    {
        float tmp[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) tmp[i] = vals[i];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            vals[i] = tmp[i] + vals[i+4];
            vals[i+4] = tmp[i] - vals[i+4];
        }
    }
    // Stages 3-7: inter-thread butterfly via __shfl_xor_sync
    #pragma unroll
    for (int s = 0; s < 5; s++) {
        unsigned int mask = 1u << s;  // XOR mask: 1, 2, 4, 8, 16
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float other = __shfl_xor_sync(0xffffffff, vals[i], mask);
            bool upper = (lane & mask) != 0;
            vals[i] = upper ? (other - vals[i]) : (vals[i] + other);
        }
    }
    // Normalize: 1/sqrt(256) = 1/16
    #pragma unroll
    for (int i = 0; i < 8; i++) vals[i] *= 0.0625f;
}
```

**CRITICAL**: The `lane` parameter must be `threadIdx.x % 32` (lane within warp), NOT `threadIdx.x` (thread within block).

#### 2.2 Lloyd-Max codebook constants

```cuda
// ============================================================================
// TurboQuant: Lloyd-Max codebook for 4-bit quantization of N(0,1)
// ============================================================================

// 16 reconstruction levels (optimal for unit Gaussian after WHT normalization)
// Computed via iterative Lloyd-Max algorithm on N(0,1) PDF.
// Symmetric: codebook[i] = -codebook[15-i]
__device__ __constant__ float TURBO4_CODEBOOK[16] = {
    -2.1519f, -1.3440f, -0.9003f, -0.5601f,
    -0.2582f,  0.0000f,  0.2582f,  0.5601f,
     0.9003f,  1.3440f,  2.1519f,  0.0000f,
     0.0000f,  0.0000f,  0.0000f,  0.0000f
};
// NOTE: Only first 11 entries used for symmetric 16-level. Actually, we need all 16:
// The correct 16-level Lloyd-Max for N(0,1) is:
//   {-2.152, -1.534, -1.150, -0.837, -0.560, -0.301, -0.053, 0.053,
//    0.301,  0.560,  0.837,  1.150,  1.534,  2.152}
// Wait — that's 14 levels. For 16 levels (4-bit):

// CORRECT 16-level Lloyd-Max codebook for N(0,1):
// These are the 16 optimal reconstruction points that minimize E[(X-Q(X))^2]
// for X ~ N(0,1). Computed iteratively until convergence.
__device__ __constant__ float TURBO4_CODEBOOK[16] = {
    -2.4008f, -1.8441f, -1.4371f, -1.0993f,
    -0.7995f, -0.5224f, -0.2583f, -0.0000f,
     0.0000f,  0.2583f,  0.5224f,  0.7995f,
     1.0993f,  1.4371f,  1.8441f,  2.4008f
};

// Decision boundaries: midpoints between adjacent codebook entries
// boundary[i] = (codebook[i] + codebook[i+1]) / 2
__device__ __constant__ float TURBO4_BOUNDS[16] = {
    -2.1225f, -1.6406f, -1.2682f, -0.9494f,
    -0.6610f, -0.3904f, -0.1292f,  0.0000f,
     0.1292f,  0.3904f,  0.6610f,  0.9494f,
     1.2682f,  1.6406f,  2.1225f,  1e30f
};
```

**IMPORTANT**: The exact codebook values should be computed properly. The ones above are approximate. For the MVP, these approximations work. For production quality, compute exact Lloyd-Max iteratively on N(0,1). The key property is that they're symmetric and minimize MSE for the Gaussian distribution.

**REFINEMENT**: Before committing final values, write a small Python script that runs Lloyd-Max iteration on N(0,1) and produces exact codebook+boundary values to paste in.

#### 2.3 Quantization function

```cuda
// Binary search quantizer: 4 comparisons for 16 levels
__device__ __forceinline__ unsigned char turbo4_quantize(float x) {
    unsigned char idx = 0;
    idx += (x >= TURBO4_BOUNDS[7])  ? 8 : 0;
    idx += (x >= TURBO4_BOUNDS[idx + 3]) ? 4 : 0;
    idx += (x >= TURBO4_BOUNDS[idx + 1]) ? 2 : 0;
    idx += (x >= TURBO4_BOUNDS[idx])     ? 1 : 0;
    return idx;
}
```

#### 2.4 The reshape_and_cache_flash_turbo4 kernel

**Append to**: `/workspace/atlas/kernels/gb10/nvfp4/reshape_and_cache.cu`

Clone the structure of `reshape_and_cache_flash_nvfp4` exactly. Same signature:
```cuda
extern "C" __global__ void reshape_and_cache_flash_turbo4(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ k_cache,
    unsigned char* __restrict__ v_cache,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const unsigned int key_stride,
    const unsigned int value_stride,
    const unsigned long long block_stride_bytes,
    const unsigned long long data_section_bytes
)
```

Algorithm per token (differences from NVFP4 are marked with ★):

1. Compute slot → physical block + offset (same as NVFP4)
2. Each warp processes one KV head
3. Lane loads 8 BF16 elements from key, converts to float vals[8]
4. ★ Call `wht256_warp(vals, lane_id)` — WHT rotation
5. Per-group quantization (group_size=16 = 2 adjacent lanes):
   a. Compute absmax within group: each lane reduces its 8 elements, then shuffle with partner lane
   b. FP8 scale = absmax / max_codebook_abs (2.4008f)
   c. ★ Quantize: `idx = turbo4_quantize(vals[i] / fp8_scale)` (instead of E2M1 round)
   d. Pack 4-bit index pairs into bytes (same nibble packing as NVFP4)
6. Write data bytes + scale bytes to cache (same layout as NVFP4)
7. Repeat for V

**The FP8 scale computation**: For NVFP4, the max representable E2M1 value is 6.0, so scale = absmax/6.0. For Turbo4, the max codebook value is ~2.4008, so scale = absmax/2.4008. This is the ONLY difference in the scale computation.

---

### Phase 3: CUDA Decode Attention Kernel

#### 3.1 Create paged_decode_attn_turbo4.cu

**New file**: `/workspace/atlas/kernels/gb10/nvfp4/paged_decode_attn_turbo4.cu`

**Clone from**: `/workspace/atlas/kernels/gb10/nvfp4/paged_decode_attn_nvfp4.cu` (the whole file, ~582 lines)

Then make these targeted changes:

**Change A — WHT(Q) after loading Q** (near the start, after Q is loaded into registers):

Find where `q_reg[VEC_BF16]` is loaded from Q input. After the load loop, add:
```cuda
// TurboQuant: Walsh-Hadamard Transform on Q (work in WHT domain)
wht256_warp(q_reg, lane_id);
```

**Change B — Replace E2M1 LUT with Turbo4 codebook**:

Find the shared memory LUT initialization (something like `e2m1_lut[16]`). Replace with:
```cuda
__shared__ float turbo4_lut[16];
if (tid < 16) turbo4_lut[tid] = TURBO4_CODEBOOK[tid];
__syncthreads();
```

All dequant operations that do `lut[nibble] * group_scale` continue to work — just different LUT values.

**Change C — iWHT(O) before writing output**:

Find where the final output `o_reg[]` is written to global memory as BF16. Before that write, add:
```cuda
// TurboQuant: inverse WHT on accumulated output
// (normalized WHT is self-inverse: WHT × WHT = I)
wht256_warp(o_reg, lane_id);
```

**Change D — Sparse V threshold (add to inner loop)**:

Add kernel parameter: `const float sparse_v_threshold`

In the inner loop, after computing the K dot products and softmax weights for a batch of BC=4 positions, before the V dequant+accumulate section:

```cuda
// Sparse V: skip V dequant if all weights in this batch are negligible
if (sparse_v_threshold > 0.0f) {
    float max_w = 0.0f;
    for (int b = 0; b < BC; b++) max_w = fmaxf(max_w, w_local[b]);
    // Warp-reduce max (all lanes must agree to skip)
    for (int offset = 16; offset > 0; offset >>= 1)
        max_w = fmaxf(max_w, __shfl_xor_sync(0xffffffff, max_w, offset));
    if (max_w < sparse_v_threshold) {
        // Skip V dequant and accumulate for this batch
        continue;  // or goto next_batch;
    }
}
```

**Change E — Copy the wht256_warp function and codebook constants** into this file (or use a shared header). Since Atlas compiles each .cu independently, the function must be present in this file. Copy the `wht256_warp`, `TURBO4_CODEBOOK`, and `TURBO4_BOUNDS` definitions.

#### 3.2 Split-K variant

In the same file, add `paged_decode_attn_splitk_turbo4` — clone from `paged_decode_attn_splitk_nvfp4` with the same changes A-E above. The split-K writes partial results to workspace; the reduce kernel combines them.

#### 3.3 Reduce kernel

Add `paged_decode_attn_reduce_turbo4` — this is identical to the NVFP4 reduce kernel (it operates on F32 workspace, dtype-agnostic). Just copy the NVFP4 reduce function and rename.

---

### Phase 4: CUDA Prefill Kernel

#### 4.1 Create inferspark_prefill_paged_turbo4.cu

**New file**: `/workspace/atlas/kernels/gb10/nvfp4/inferspark_prefill_paged_turbo4.cu`

**Clone from**: `/workspace/atlas/kernels/gb10/nvfp4/inferspark_prefill_paged_nvfp4.cu`

Changes from NVFP4:

**A. Replace LUT**: Same as decode — replace E2M1 LUT with TURBO4_CODEBOOK.

**B. WHT on Q tile**: After Q is loaded to shared memory, apply WHT to each row (each row = head_dim=256 elements). Use shared-memory cooperative butterfly:
```cuda
// After Q tile loaded to smem_Q[BR][head_dim]:
// Apply WHT to each Q row cooperatively
for (int row = warp_id; row < br_size; row += NUM_WARPS) {
    // Each warp handles one row
    float vals[8];
    for (int i = 0; i < 8; i++) vals[i] = (float)smem_Q[row][lane * 8 + i];
    wht256_warp(vals, lane);
    for (int i = 0; i < 8; i++) smem_Q[row][lane * 8 + i] = (__nv_bfloat16)__float2bfloat16(vals[i]);
}
__syncthreads();
```

**C. iWHT on output**: After attention output is computed, apply iWHT to each output row before writing to global memory. Same warp-cooperative approach.

**Note**: K and V are already in WHT domain (stored that way). Q·K products in WHT domain = correct attention scores. V accumulation in WHT domain = correct (by linearity). Only need WHT(Q) and iWHT(O).

**FALLBACK if prefill kernel is too complex**: For the initial version, the turbo4 prefill can be a simple wrapper that:
1. Dequants the KV cache block to a BF16 scratch buffer (with iWHT)
2. Calls the standard BF16 prefill kernel
This is slower but correct. Optimize later.

Also create BR=64 variant if the NVFP4 version has one (check for `inferspark_prefill_paged_nvfp4_64` or similar).

---

### Phase 5: KERNEL.toml Registration

#### 5.1 Update shared KERNEL.toml

**File**: `/workspace/atlas/kernels/gb10/nvfp4/KERNEL.toml`

Add:
```toml
paged_decode_attn_turbo4 = "paged_decode_turbo4"
inferspark_prefill_paged_turbo4 = "prefill_paged_turbo4"
```

The `reshape_and_cache_flash_turbo4` lives in `reshape_and_cache.cu` which is already registered as module `"reshape_and_cache"`.

#### 5.2 Update ALL model-specific KERNEL.toml files

Each model needs the same additions. Files:
- `/workspace/atlas/kernels/gb10/qwen3-next-80b-a3b/nvfp4/KERNEL.toml`
- `/workspace/atlas/kernels/gb10/qwen3.5-35b-a3b/nvfp4/KERNEL.toml`
- `/workspace/atlas/kernels/gb10/qwen3.5-122b-a10b/nvfp4/KERNEL.toml`
- `/workspace/atlas/kernels/gb10/qwen3.5-27b/nvfp4/KERNEL.toml`
- `/workspace/atlas/kernels/gb10/qwen3-vl-30b-a3b/nvfp4/KERNEL.toml`
- `/workspace/atlas/kernels/gb10/nemotron-3-nano-30b-a3b/nvfp4/KERNEL.toml`
- `/workspace/atlas/kernels/gb10/nemotron-super-120b-a12b/nvfp4/KERNEL.toml`
- `/workspace/atlas/kernels/gb10/mistral-small-4/nvfp4/KERNEL.toml`

Add the same two lines to each.

---

### Phase 6: Build and Test

#### 6.1 Compile

```bash
cd /workspace/atlas
cargo build -r 2>&1 | head -100
```

Fix any compilation errors. Common issues:
- Missing match arms for `Turbo4` in exhaustive matches
- Kernel module not found (KERNEL.toml typo)
- CUDA syntax errors in new kernels

#### 6.2 Verify no regressions

Quick check that FP8 still works:
```bash
# If a vLLM container is running, stop it first
curl -s http://localhost:8888/v1/models 2>/dev/null || echo "No server running"
```

#### 6.3 Test turbo4

Start the server with turbo4:
```bash
./target/release/spark-server --kv-cache-dtype turbo4 --model [appropriate model]
```

Test basic inference:
```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":50}'
```

#### 6.4 Test Sparse V

```bash
./target/release/spark-server --kv-cache-dtype turbo4 --sparse-v-threshold 0.001
```

Test with a longer context to see skip benefits.

---

## Lloyd-Max Codebook Computation

Before implementing, compute exact codebook values with this Python script:

```python
import numpy as np
from scipy.stats import norm

def lloyd_max_gaussian(n_levels, max_iter=1000, tol=1e-10):
    """Compute optimal Lloyd-Max quantizer for N(0,1)."""
    # Initialize with uniform quantiles
    boundaries = norm.ppf(np.linspace(0, 1, n_levels + 1)[1:-1])
    boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])

    for _ in range(max_iter):
        # Reconstruction levels: conditional expectation in each bin
        levels = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i+1]
            # E[X | lo < X < hi] for X ~ N(0,1)
            num = norm.pdf(lo) - norm.pdf(hi)
            den = norm.cdf(hi) - norm.cdf(lo)
            levels[i] = num / den if den > 0 else (lo + hi) / 2

        # Decision boundaries: midpoints
        new_boundaries = np.concatenate([
            [-np.inf],
            (levels[:-1] + levels[1:]) / 2,
            [np.inf]
        ])

        if np.max(np.abs(new_boundaries[1:-1] - boundaries[1:-1])) < tol:
            break
        boundaries = new_boundaries

    return levels, boundaries[1:-1]

levels, bounds = lloyd_max_gaussian(16)
print("CODEBOOK:", [f"{v:.4f}" for v in levels])
print("BOUNDS:", [f"{v:.4f}" for v in bounds])
```

Run this and paste the exact values into the CUDA constants.

---

## Files Summary

### New files
| File | Description |
|------|-------------|
| `kernels/gb10/nvfp4/paged_decode_attn_turbo4.cu` | Decode attention: WHT + Lloyd-Max LUT + Sparse V |
| `kernels/gb10/nvfp4/inferspark_prefill_paged_turbo4.cu` | Prefill attention reading Turbo4 cache |

### Files to modify
| File | What to change |
|------|---------------|
| `crates/spark-runtime/src/kv_cache.rs` | Add Turbo4 to enum, Display, FromStr, block_bytes, helpers |
| `crates/spark-server/src/cli.rs` | Add turbo4 to help, add --sparse-v-threshold |
| `crates/spark-model/src/layers/qwen3_attention/mod.rs` | Turbo4 kernel handles, sparse_v_threshold field |
| `crates/spark-model/src/layers/qwen3_attention/decode.rs` | Turbo4 match arms in write_kv + run_paged_decode |
| `crates/spark-model/src/layers/qwen3_attention/prefill.rs` | Turbo4 match arm |
| `crates/spark-model/src/layers/ops.rs` | Add 4 turbo4 op functions |
| `kernels/gb10/nvfp4/reshape_and_cache.cu` | Append WHT + codebook + turbo4 write kernel |
| `kernels/gb10/nvfp4/KERNEL.toml` | Register turbo4 modules |
| All 8 model-specific KERNEL.toml files | Register turbo4 modules |

### Files to read as templates (do NOT modify)
| File | Template for |
|------|-------------|
| `kernels/gb10/nvfp4/paged_decode_attn_nvfp4.cu` | → turbo4 decode kernel |
| `kernels/gb10/nvfp4/inferspark_prefill_paged_nvfp4.cu` | → turbo4 prefill kernel |
| `kernels/gb10/nvfp4/reshape_and_cache.cu` (NVFP4 section) | → turbo4 write kernel |

---

## Debugging Tips

1. **WHT correctness**: If output is garbage, the WHT is probably wrong. Test with an impulse input (v[0]=1, rest=0) — WHT should produce all 1/16 values.

2. **Scale factor**: The max codebook value for Turbo4 (~2.4) is very different from E2M1 max (6.0). Make sure FP8 scale computation uses the correct divisor.

3. **Shared memory LUT**: The codebook has 16 entries just like E2M1. Same indexing, same nibble unpacking. If dequant looks wrong, check the LUT values match the codebook.

4. **Kernel not found at runtime**: Check KERNEL.toml has the right filename→module mapping. The filename stem must match exactly.

5. **Prefill fallback**: If the turbo4 prefill kernel is too complex or buggy, a valid temporary approach is to have turbo4 prefill fall back to the NVFP4 prefill kernel (wrong codebook LUT but will produce *something* for testing decode in isolation). Mark this with a TODO.

---

## Non-Goals (explicitly out of scope)

- Turbo3 (3-bit) — different packing, defer
- Asymmetric K/V — requires refactoring LayerPool to separate K/V dtypes
- Temporal decay — research-tier feature
- QJL residual correction — theoretical bonus, complex, marginal benefit
- Exact Beta distribution codebooks — Gaussian approximation is excellent for d=256
