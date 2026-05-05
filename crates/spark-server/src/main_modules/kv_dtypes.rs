// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer KV cache dtype vector construction.

/// Build per-attention-layer KV cache dtype vector.
///
/// When `high_precision_layers` is 0, returns an empty vec (all layers use uniform dtype).
/// When non-zero, the first N and last N attention layers use BF16; middle layers use
/// the base `kv_dtype`. If `kv_dtype` is already BF16, returns empty vec (no benefit).
pub(crate) fn build_layer_kv_dtypes(
    kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    num_attention_layers: usize,
    high_precision_layers: usize,
) -> Vec<spark_runtime::kv_cache::KvCacheDtype> {
    use spark_runtime::kv_cache::KvCacheDtype;

    if high_precision_layers == 0 || kv_dtype == KvCacheDtype::Bf16 {
        return vec![];
    }

    // Clamp: if 2*N >= num_attention_layers, all layers become BF16.
    let hp = high_precision_layers.min(num_attention_layers);
    let mut dtypes = vec![kv_dtype; num_attention_layers];

    for i in 0..hp.min(num_attention_layers) {
        dtypes[i] = KvCacheDtype::Bf16;
    }
    for i in num_attention_layers.saturating_sub(hp)..num_attention_layers {
        dtypes[i] = KvCacheDtype::Bf16;
    }

    let hp_count = dtypes.iter().filter(|d| **d == KvCacheDtype::Bf16).count();
    tracing::info!(
        "Selective high-precision KV cache: {}/{} attention layers at BF16, rest at {}",
        hp_count,
        num_attention_layers,
        kv_dtype,
    );

    dtypes
}
