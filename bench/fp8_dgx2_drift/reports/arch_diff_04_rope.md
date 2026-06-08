# Arch-Diff #04 — RoPE / MRoPE-Interleaved (vLLM vs Atlas)

Scope: Qwen3.6-A3B MRoPE-interleaved (`mrope_section=[11,11,10]`, `rotary_dim=64`, `theta=10_000_000`). For text-only inputs `pos_t == pos_h == pos_w`, so the kernel collapses to scalar RoPE.

## Cross-engine function map

| Stage | vLLM | Atlas |
|---|---|---|
| inv_freq | `RotaryEmbeddingBase._compute_inv_freq` (rotary_embedding/base.py:55) — FP32 `1 / base ** (arange(0,rd,2)/rd)` | computed inline per call in FP64: `1.0 / pow((double)theta, 2*pair_idx/rotary_dim)` (rope.cu:72-73, rope_mrope_interleaved.cu:78-79) |
| cos/sin table | precomputed `cos_sin_cache` over `max_position*4` positions, FP32 then cast to query dtype (base.py:46-50, mrope.py:237) | none — recomputed every call per (pair_idx, abs_pos) |
| Q/K storage during rotation | `apply_token_rotary_embedding` reads/writes `scalar_t` (BF16) directly (`x*cos - y*sin` in BF16) (pos_encoding_kernels.cu:30-33) | reads BF16 → FP32, rotates in FP32, rounds back to BF16 (rope.cu:95-104, rope_mrope_interleaved.cu:99-106) |
| Q/K mutation | in-place (pos_encoding_kernels.cu:32-33) | in-place on `k_contiguous` (paged.rs:351-368) |
| MRoPE section dispatch (mm) | per-axis masks on `cos_offsets`; interleaved: `(off%3==1)→H`, `(off%3==2)→W`, else T (mrope.py:61-64) | `pair_idx % 3 → {0:T,1:H,2:W}` selects abs_pos lookup table (rope_mrope_interleaved.cu:70-75) |
| Text-only MRoPE | `positions.ndim==1` branch bypasses MRoPE entirely → scalar RoPE via `apply_rotary_emb_dispatch` (mrope.py:347-358) | always runs MRoPE kernel; stacks `pos_h = pos_w = pos_t` upstream (upload_meta.rs:137-146) |

## Findings

### 1. cos/sin precision (FP64 vs FP32 inv_freq)

Atlas: FP64 `pow(theta, 2k/rd)` per call (rope.cu:72-73). At theta=1e7, k=31, rd=64, base inv_freq ≈ 1.6e-7; FP32 powf has ~1e-6 rel-err → angle drift at pos=64k ≈ 1e-8 rad — negligible. Comment at rope.cu:69-71 cites 30k-pos drift, also ~5e-9 rad absolute. FP64 `pow` is **not load-bearing** for Qwen3.6.

vLLM caches cos/sin FP32 then casts to BF16 for kernel use (`_match_cos_sin_cache_dtype`, base.py:80-87) — only 7 bits of cos/sin mantissa survive. **Atlas is more precise here.**

### 2. Rotation arithmetic — divergent MAC dtype

Both use rotate_half / NeoX, pair `(d0, d0+rd/2)`, `y0 = x0 cos − x1 sin`, `y1 = x1 cos + x0 sin` (rope.cu:99-100, rope_mrope_interleaved.cu:102-103; pos_encoding_kernels.cu:30-33).

- **vLLM**: `scalar_t` is BF16; MACs in BF16 (no upcast). Two MACs ≈ 2 ulp BF16 ≈ 8e-3.
- **Atlas**: explicit FP32 upcast, MACs in FP32, single BF16 round at store (rope.cu:95-104).

Atlas has **strictly tighter K rounding** post-RoPE. Not a bug.

### 3. In-place K mutation before KV-cache write

Both rotate K in place in BF16; quant→FP8 happens downstream of the rotated BF16 buffer. No FP32-scratch path on either side.

**FP8 drift relevance**: FP8 E4M3 mantissa (2^-3) dominates the BF16 RoPE rounding (2^-7) by ~16×. **Not a candidate root cause for the FP8-vs-BF16 drift under investigation.**

### 4. MRoPE section assignment — equivalent under text-only

Atlas: `section = pair_idx % 3` (rope_mrope_interleaved.cu:70-75). Pair counts under modulo: indices 0,3,…,30 → T (11); 1,4,…,31 → H (11); 2,5,…,29 → W (10).

vLLM `[11,11,10]`: `(off%3==1) & (off≤33)→H`; `(off%3==2) & (off≤30)→W`; rest → T. For `half_rd=32`: T={0,3,…,30, 31}=12, H={1,4,…,31}=11, W={2,5,…,29}=10.

**Discrepancy on pair counts:** pair 31 (highest freq) is **T in vLLM, H in Atlas**. For text-only `pos_t == pos_h`, both compute the same `abs_pos * freq` → bit-identical output. **No drift on text-only.** For multimodal (T≠H≠W) the boundary pair diverges — flagged for future MM coverage.

### 5. Section-sum invariant — Atlas does not validate

vLLM asserts `sum(mrope_section) == rotary_dim // 2` (mrope.py:250). Atlas has no equivalent check; the modulo kernel ignores config. **Low-risk lint.**

## Bug flags

- **None blocking the FP8 drift investigation.** RoPE is bit-equivalent across engines for text-only Qwen3.6.
- **FLAG (multimodal, future):** for `mrope_section=[a,b,c]` with `a≠b≠c`, Atlas's pure-modulo section assignment will disagree with vLLM's masked assignment on the boundary pairs. Re-validate before shipping any MM path on Qwen3.6.
- **Lint:** Atlas should `debug_assert!(sum(mrope_section) == rotary_dim/2)` at config load (no current file; suggest `model/config.rs` at MRoPE deserialize site).
- **Perf note (not bug):** FP64 `pow` per pair per token per layer (rope.cu:73) — for prefill of 32k tokens × 32 layers × 32 pairs ≈ 33M FP64 `pow` calls/request. Replaceable with a per-rank precomputed FP32 `inv_freq[32]` table loaded once (vLLM model). Same precision in practice for `rd=64`.

## Files cited

- `/workspace/atlas-mtp/kernels/gb10/common/rope.cu:29-105`
- `/workspace/atlas-mtp/kernels/gb10/common/rope_mrope_interleaved.cu:34-107`
- `/workspace/atlas-mtp/crates/spark-model/src/layers/qwen3_attention/prefill/paged.rs:258-368`
- `/workspace/atlas-mtp/crates/spark-model/src/model/trait_impl/prefill_b/upload_meta.rs:102-148`
- `/home/nologik/vllm/vllm/vllm/model_executor/layers/rotary_embedding/base.py:55-78, 104-164`
- `/home/nologik/vllm/vllm/vllm/model_executor/layers/rotary_embedding/mrope.py:15-188, 247-358`
- `/home/nologik/vllm/vllm/vllm/model_executor/layers/rotary_embedding/common.py:22-70`
- `/home/nologik/vllm/vllm/csrc/pos_encoding_kernels.cu:10-100`
