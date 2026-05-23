// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer KV cache dtype vector construction.

/// Build per-attention-layer KV cache dtype vector.
///
/// When `high_precision_layers` is 0, returns an empty vec (all layers use uniform dtype).
/// When non-zero, the first N and last N attention layers use `boundary_dtype` (default
/// BF16); middle layers use the base `kv_dtype`. Per TQ+ LA-V7 Mode 7 (Tom upstream): a
/// flexible boundary policy lets you mix e.g. middle=Turbo2 + boundary=Fp8 instead of
/// the rigid middle=Turbo2 + boundary=BF16. Returns empty vec if `boundary_dtype` ==
/// `kv_dtype` (no benefit) or `high_precision_layers` == 0.
pub(crate) fn build_layer_kv_dtypes(
    kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    num_attention_layers: usize,
    high_precision_layers: usize,
    boundary_dtype: spark_runtime::kv_cache::KvCacheDtype,
) -> Vec<spark_runtime::kv_cache::KvCacheDtype> {
    if high_precision_layers == 0 || kv_dtype == boundary_dtype {
        return vec![];
    }

    let hp = high_precision_layers.min(num_attention_layers);
    let mut dtypes = vec![kv_dtype; num_attention_layers];

    for i in 0..hp.min(num_attention_layers) {
        dtypes[i] = boundary_dtype;
    }
    for i in num_attention_layers.saturating_sub(hp)..num_attention_layers {
        dtypes[i] = boundary_dtype;
    }

    let hp_count = dtypes.iter().filter(|d| **d == boundary_dtype).count();
    tracing::info!(
        "Selective boundary KV cache: {}/{} attention layers at {}, rest at {}",
        hp_count,
        num_attention_layers,
        boundary_dtype,
        kv_dtype,
    );

    dtypes
}
