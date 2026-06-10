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
/// Auto high-precision-layer count for turbo KV dtypes when the user passed
/// `--kv-high-precision-layers 0` (the default).
///
/// Returns `None` for dtypes that need no boundary layers (bf16/fp8/nvfp4).
/// Baseline formula: ceil(num_attn / 3), floor 2 — keeps accumulated turbo
/// quant error tractable as attention-layer count grows.
///
/// Turbo2 (symmetric 2-bit K) and Bf16KTurbo3V get ceil(4 * num_attn / 5),
/// floor 4: on the GB10 flagship (Qwen3.6-35B-A3B-FP8, 10 attention layers)
/// both score 0/5-0/10 on the agentic webserver suite at the baseline auto of
/// 4 (turbo2: tool-call envelope collapse; bf16k_turbo3v: port-instruction
/// drift) and recover to 5/5 at 8, which this formula reproduces.
pub(crate) fn auto_high_precision_layers(
    kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    num_attention_layers: usize,
) -> Option<usize> {
    use spark_runtime::kv_cache::KvCacheDtype as D;
    match kv_dtype {
        D::Bf16 | D::Fp8 | D::Nvfp4 => None,
        D::Turbo2 | D::Bf16KTurbo3V => Some(((num_attention_layers * 4).div_ceil(5)).max(4)),
        D::Turbo3
        | D::Turbo4
        | D::Turbo8
        | D::Turbo4KTurbo3V
        | D::Turbo4KTurbo8V
        | D::Turbo3KTurbo8V
        | D::Bf16KTurbo4V
        | D::Fp8KTurbo4V
        | D::Fp8KTurbo3V
        | D::Bf16KTurbo2V
        | D::Fp8KTurbo2V => Some(((num_attention_layers as f32 / 3.0).ceil() as usize).max(2)),
    }
}

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
