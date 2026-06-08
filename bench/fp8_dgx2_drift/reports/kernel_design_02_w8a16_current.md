# W8A16 Current Kernel Anatomy — BF16-narrowing-in-smem bug

Trace of where the W8A16 path collapses precision and the smallest change to
match vLLM's W8A8 + FP32-epilogue shape. Files:

- `kernels/gb10/common/w8a16_gemm.cu`, `…/w8a16_gemm_t.cu`, `…/w8a16_gemv.cu`
- `crates/spark-model/src/layers/ops/gemm_quant.rs`

## 1. `w8a16_gemm` (non-transposed)

- Tile: `BM × BN × BK = M_TILE=64 × N_TILE=64 × K_STEP=16` (`w8a16_gemm.cu:19-22`).
- Grid `(ceil(N/64), ceil(M/64), 1)`, Block `(128,1,1)` — 4 warps/CTA, each warp owns 16 rows of M.
- HBM load:
  - A: per-thread strided load `(M_TILE*K_STEP)/128 = 8` BF16 elements → smem (`w8a16_gemm.cu:183-192`).
  - B: each thread reads 8 FP8 bytes, **scalar `B[gn*K + gk]` with stride `K`** (`w8a16_gemm.cu:208`). Not coalesced (N is the slow axis).
  - Block scale: BF16, indexed `block_scale[n_block * k_blocks + k_block]`, read as BF16 and converted with `__bfloat162float` (`w8a16_gemm.cu:212`).
- smem layout: `smem_A[64][16+PAD]`, `smem_B[16][64+PAD]`, both `__nv_bfloat16` (`w8a16_gemm.cu:168-169`).
- FP8 → FP32 LUT lookup: `E4M3_LUT[weight_byte] * scale` at `w8a16_gemm.cu:214` (FP32 multiply).
- **BF16 narrowing happens at `w8a16_gemm.cu:215`**: `smem_B[k][n] = __float2bfloat16(dequant_val);`. Every dequanted weight loses ~8 bits of mantissa before it ever reaches an MMA.
- MMA: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`w8a16_gemm.cu:131-143`). Reads `smem_A`/`smem_B` as packed pairs of BF16, accumulates FP32. So accumulator type is OK; precision loss is purely on the B operand.
- Scale application: **per-element early multiply** (line 214) on the dequanted FP32 value, *then* discarded by the BF16 store. Scales are not folded into the accumulator.
- Epilogue: per-thread `acc[n_tile][k] = float` → `__float2bfloat16(...)` on store to `C` (`w8a16_gemm.cu:236-239`). FP32 accumulator → BF16 output, standard.

## 2. `w8a16_gemm_t` (transposed)

Same shape as non-transposed, two changes:

- Macros declare `M_TILE=64 × N_TILE=128 × K_STEP=32` (`w8a16_gemm_t.cu:19-21`), **but the body actually loads a 16×64 B-tile** (`smem_B[16][64+PAD]` line 167; loop `k_base += 16` line 179; `cta_n` advances by 64 line 160). The `N_TILE=128`/`K_STEP=32` macros are dead — comment-only.
- HBM load is coalesced: `B_t[gk * N + gn]` (`w8a16_gemm_t.cu:209`) — adjacent threads, adjacent N bytes. Precision pipeline identical to non-transposed.
- Block scale read BF16 → FP32 at `w8a16_gemm_t.cu:214`, LUT × scale at line 216, **same BF16 narrowing at line 217**: `smem_B[k][n] = __float2bfloat16(dequant_val);`. Same MMA (`:127-139`), same epilogue (`:238-241`).

## 3. Rust dispatch + callsites

Wrappers in `crates/spark-model/src/layers/ops/gemm_quant.rs`:
- `w8a16_gemm` (`gemm_quant.rs:201-224`), 8 kernel args: `A, B, block_scale, C, M, N, K`.
- `w8a16_gemm_t` (`gemm_quant.rs:344-367`), same signature except `B_t, block_scale_t`.

Callsites per transformer layer (Qwen3.6 FP8):

| Module | File | Callsite count |
|--------|------|----------------|
| Q/K/V proj (paged prefill) | `qwen3_attention/prefill/paged_qkv.rs:130,144` | 3 (×3 projs called separately) |
| Q/K/V proj (cache-skip prefill) | `qwen3_attention/prefill/cache_skip_qkv.rs:150,164` | 3 alt path |
| O proj | `qwen3_attention/prefill/paged_oproj.rs:29,45` | 1 |
| MoE shared gate / up / down | `moe/forward_prefill_fp8.rs:51,63,86` | 3 |
| Unified dispatch (covers other proj sites) | `ops/quant_dispatch.rs:71` | n/a (delegates) |

~7 W8A16 launches per FP8 layer per prefill step (Q,K,V,O + 3 shared-expert) × ~48 layers ≈ 340 launches per prefill — every one narrows weights to BF16 before MMA.

## 4. Scale storage confirmation (`linear_attn_arms.rs:111`)

`scale_row_bytes = scale_cols * 2` at `linear_attn_arms.rs:111`, with `scale_cols = h / BS` BF16 entries per row (`:110`). Comment block at `:92-96` confirms "BF16". Kernels read as BF16 and immediately upcast: `__bfloat162float(block_scale[...])` (`w8a16_gemm.cu:212`, `w8a16_gemm_t.cu:214`, `w8a16_gemv.cu:141`). Scale storage is consistent — **no layout change needed**.

The GEMV decode kernel already does the right thing: `LUT[byte] * scale` in FP32 and accumulates in FP32 (`w8a16_gemv.cu:160-174`) — never narrows to BF16. This matches the observed pattern that decode drift is much smaller than prefill drift: the bug is prefill-only.

## 5. Minimum change

The MMA primitive in use — `mma.sync.m16n8k16.f32.bf16.bf16.f32` — has no FP8 input variant on sm_121 (project_fp4_mma_gb10 negative finding still holds for FP8 K-input). Two viable shapes:

(a) **Keep MMA as bf16.bf16, but apply block scales in the FP32 accumulator** (vLLM-style "W8A8 with FP32 epilogue"). Concretely:
   - Treat each `K_STEP=16` segment as belonging to one `(n_block, k_block)` pair. With `FP8_BLOCK=128` and `K_STEP=16` the whole tile lives inside a single scale block in K, and along N each 8-thread group also lives in one N-block (`N_TILE=64 ≤ 128`).
   - At dequant: store `smem_B[k][n] = __float2bfloat16(E4M3_LUT[byte])` **without** multiplying by `scale`. The LUT range fits BF16 cleanly (LUT values are exact dyadic rationals ≤ 448).
   - Accumulate normally with bf16 MMA into `acc[8][4]` (FP32).
   - In the epilogue, scale: `acc[n_tile][k] *= scale_for(cta_n + frag_col, cta_m + frag_row_k_chunk)` once per K-block boundary. Since `K_STEP=16 < FP8_BLOCK=128`, this means scale is only re-loaded every 8th K-step, and `acc` accumulates `BF16(LUT_byte)·BF16(A)` in FP32 — i.e. the same math vLLM does for W8A8.

(b) Full W8A8: MMA over `f8e4m3.f8e4m3.f32`. Not available on sm_121.

(a) is **strictly a kernel-internal change**. The Rust dispatch signature, scale storage layout, weight transpose layout, and all 7 per-layer callsites stay byte-identical. The only behavioral diff is that the BF16 cast of `dequant_val` moves from per-element-pre-MMA (`w8a16_gemm.cu:215`) to per-element-pre-MMA-without-scale, and a scaled accumulator multiply is inserted before the final BF16 store (`w8a16_gemm.cu:236-239`).

### Patch-in-place vs greenfield — recommendation

**Patch in place.** The diff is roughly:
1. Drop `* scale` from line 214 (and 216 in `_t`).
2. Inside the K-loop, hoist `scale` for the current `(cta_n N-block, k_base K-block)` pair into a per-warp FP32 (or per-thread register array of 8 entries for the 8 N-fragments at `cta_n + n_tile*8`).
3. When `k_base` crosses an FP8_BLOCK boundary (i.e. `k_base % FP8_BLOCK == 0` and `k_base > 0`), pre-scale `acc` by `prev_scale / current_scale` — or simpler: keep a parallel `acc_scaled[8][4]` and flush `acc` into it whenever the scale changes, then zero `acc`.
4. Final epilogue: store from `acc_scaled` (already scaled) via `__float2bfloat16` exactly as today.

Total touched lines: ≤ 30 in each of `w8a16_gemm.cu` and `w8a16_gemm_t.cu`. No Rust code changes (the kernel handles `block_scale` internally; the buffer ABI is unchanged). The 7 callsites per layer continue working unmodified.

A greenfield kernel would force migration of 7 callsites × 2 prefill paths × ~5 model loaders, plus a parallel Rust wrapper, with no precision gain over (a). Save greenfield for the eventual FP8-native MMA when it ships on a future SM.
