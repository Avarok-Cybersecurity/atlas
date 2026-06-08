# Arch Diff #05 — Paged KV Cache Write (post-RoPE)

Scope: write of post-RoPE K and un-rotated V into the paged KV cache at
positions identified by `slot_mapping`. Comparing vLLM upstream
(`/home/nologik/vllm/vllm/csrc/cache_kernels.cu`) vs Atlas
(`/workspace/atlas-mtp/kernels/gb10/common/`).

---

## 1. BF16 cache — pure memcopy?

Both engines: **yes, pure memcopy with no arithmetic.**

- vLLM `reshape_and_cache_flash_kernel` (cache_kernels.cu:286-358). When
  `kv_dt == kAuto`, `CopyWithScaleOp` collapses to `dst = static_cast<OutT>(src)`
  (line 218); since `cache_t == scalar_t == __nv_bfloat16`, this is a
  bit-preserving copy. Vectorized via `vectorize_with_alignment<VEC_SIZE=8>`
  (line 327, 330).
- Atlas `reshape_and_cache_flash` (reshape_and_cache.cu:67-125). Pure
  `uint2` memcpy (4×BF16 per thread, line 113-114). No arithmetic on the
  BF16 path at all.

Conclusion: BF16 path is **bit-identical between engines**.

---

## 2. FP8 cache — scale semantics

Both engines: **per-tensor scalar scale, divide-by-scale convention,
SATFINITE E4M3.** Same math, slightly different rounding path.

- vLLM `scaled_vec_conversion<uint8_t, __nv_bfloat16>`
  (`csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:480-490`):
  `__nv_cvt_float_to_fp8(__bfloat162float(a) / scale, __NV_SATFINITE, fp8_type)`.
  Scale is passed as `const float*` (pointer to a 1-element tensor on the
  GPU — `layer._k_scale`, `layer._v_scale`), dereferenced once at kernel
  start (cache_kernels.cu:319-320).
- Atlas `reshape_and_cache_flash_fp8` (reshape_and_cache.cu:152-215):
  `__nv_cvt_float2_to_fp8x2((v0*inv_scale, v1*inv_scale),
  __NV_SATFINITE, __NV_E4M3)` (lines 148-149), with
  `inv_scale = 1.0f / k_scale` precomputed (line 189). Scale is passed
  by **value** as a `float` kernel arg from the Rust dispatch
  (`effective_fp8_scales` returns `(f32, f32)` in decode.rs:19).

**Numerical divergence flag:**

- vLLM does `bf16 → f32 → /scale → fp8`. One multiply equivalent (`/`),
  one cast.
- Atlas does `bf16 → f32 → *inv_scale → fp8`. One multiply, one cast.
  Precomputed reciprocal `1/scale` introduces ~1 ULP of error in
  `inv_scale` vs vLLM's per-element division.
- Saturation, rounding mode, and clamp range are identical
  (`__NV_SATFINITE`, `__NV_E4M3`, ±448).
- Both write the post-RoPE BF16 source K (rounded to BF16 first, then
  re-cast to FP32 inside the cache kernel). **Neither carries K in FP32
  across the RoPE→write boundary.**

`fused_k_norm_rope_cache_write_fp8` (fused_k_norm_rope_cache.cu:252-338)
is Atlas-only and *does* skip the BF16 intermediate: it keeps K in FP32
through norm+RoPE and rounds **once** to FP8 at the write
(`out_val * inv_scale → __nv_cvt_float_to_fp8`, line 336-337). This is a
**vLLM-vs-Atlas divergence in Atlas's favor at L35-L39** (per the file's
docstring, lines 10-16). vLLM lacks this fusion — its K goes BF16 (post
RoPE) → FP32 (in cache kernel) → FP8.

---

## 3. slot_mapping arithmetic

Identical:

```
block_idx    = slot / block_size
block_offset = slot % block_size
slot < 0     → skip (padding)
```

- vLLM cache_kernels.cu:302-303.
- Atlas reshape_and_cache.cu:85-86 (BF16), :172-173 (FP8).

Both treat `slot_mapping` as `int64`. Both skip `slot < 0` early.

---

## 4. Page layout

Both engines: **NHD** = `[num_blocks, block_size, num_kv_heads, head_dim]`.

- vLLM: `get_kv_cache_shape` returns `(2, num_blocks, block_size,
  num_kv_heads, head_size)` (flash_attn.py:121). The leading `2`
  is the K/V split; `kv_cache.unbind(0)` gives `key_cache, value_cache`
  each `[num_blocks, block_size, num_kv_heads, head_size]`. NHD
  detected in the kernel via `head_stride == head_size`
  (cache_kernels.cu:317), then `n_elems = num_heads * head_size`
  contiguous write (line 327).
- Atlas: comment at reshape_and_cache.cu:12-13 declares
  `[num_blocks, block_size, num_kv_heads, head_dim]`. Address math
  (line 96-100): `block_idx * (block_size * num_kv_heads * head_dim) +
  block_offset * (num_kv_heads * head_dim)`. **Match.**

---

## 5. K and V pool storage

**Divergence — different at the allocation level, same at the kernel
interface level.**

- vLLM: single contiguous `kv_cache` tensor of shape
  `(2, num_blocks, block_size, num_kv_heads, head_size)`; `unbind(0)`
  produces two views into the same underlying allocation. K and V live
  in adjacent halves of one buffer (flash_attn.py:791).
- Atlas: two **separate** allocations `k_pool` and `v_pool` per layer
  (`crates/spark-runtime/src/kv_cache/paged_impl.rs:20-25`), accessed
  via `k_pool_ptr` / `v_pool_ptr` (paged_impl.rs:170-176). Pointers
  passed independently to `reshape_and_cache_flash`.

**Functional consequence:** none for correctness of the write itself —
both kernels receive two distinct `cache_t*` pointers and do not assume
contiguity between K and V. Only matters for memory-fragmentation /
allocator behaviour.

---

## 6. Non-paged variant

Atlas `kv_cache_append` (kv_cache_append.cu:20-57) is the contiguous
(non-paged) write for legacy flat KV layout. Pure BF16 memcpy. vLLM v1
has no analogue — always paged. Gated upstream by `PagedKvCache`
selection.

---

## Bugs flagged

**None functionally critical.** Two micro-precision notes:

1. Atlas precomputes `inv_scale = 1/scale` (reshape_and_cache.cu:189)
   then multiplies; vLLM divides per element (quant_utils.cuh:487).
   ULP-level delta only — not the L35-L39 cliff source.
2. Atlas's fused FP8 K-writer (fused_k_norm_rope_cache.cu:252-338) is
   **more precise than vLLM** for K: one rounding (FP32→FP8) vs vLLM's
   two (FP32→BF16 post-RoPE, then BF16→FP32→FP8 in cache). V matches
   vLLM (no RoPE on V).

No structural write-path bug found. Drift investigation should move to
the **attention read path** (K/V dequant + GEMM accumulator precision)
or scale calibration drift in `effective_fp8_scales`
(decode.rs:19, observer at write_kv_cache.rs:131-133).
