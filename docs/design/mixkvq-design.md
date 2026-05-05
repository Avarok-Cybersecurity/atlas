# MixKVQ Simplified: Per-Head Mixed-Precision KV Cache

Date: 2026-03-16
Status: Design phase

## Concept

Instead of uniform KV cache precision across all heads, assign higher precision
to attention heads that show higher activation variance (more information-dense).

## Simplified Algorithm

1. During a warmup phase (first 256 tokens), compute per-head variance of K values
2. Rank heads by variance: high-variance heads are more sensitive to quantization
3. Top-K% heads (e.g., top 25%) get BF16 KV cache
4. Remaining heads get NVFP4 KV cache
5. This is a per-head decision, not per-channel (simpler, no kernel changes needed)

## Implementation

### Per-Head Dtype Assignment
- After warmup, each head gets a dtype: BF16 or NVFP4
- The reshape_and_cache kernel already processes per-head — just need to
  conditionally skip quantization for BF16 heads
- The paged decode attention kernel reads K/V per-head — needs to know the dtype

### Memory Impact
- With 25% heads at BF16: memory = 0.75 * nvfp4_size + 0.25 * bf16_size
- NVFP4: ~0.5625 bytes/element (4-bit + scales)
- BF16: 2 bytes/element
- Mixed: 0.75 * 0.5625 + 0.25 * 2 = 0.922 bytes/element (~64% more than pure NVFP4)
- Still 54% less than pure BF16

### Kernel Changes Needed
- reshape_and_cache: per-head dtype flag (skip quantization for BF16 heads)
- paged_decode_attention: per-head dtype flag (skip dequant for BF16 heads)
- Both already iterate per-head — the change is a conditional in the inner loop

### Files
- `crates/spark-runtime/src/kv_cache.rs` — per-head dtype metadata
- `crates/spark-model/src/layers/qwen3_attention.rs` — head variance tracking
- `kernels/gb10/nvfp4/reshape_and_cache.cu` — per-head conditional quantization
- `kernels/gb10/nvfp4/paged_decode_attention.cu` — per-head conditional dequant
