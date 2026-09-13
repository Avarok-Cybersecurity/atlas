// SPDX-License-Identifier: AGPL-3.0-only

//! Row-wise FP8 GDN prefill BF16-weight slab sizing, split out of `sizes.rs`
//! (≤500 LoC cap). Env-gated — 0 (→ NULL) unless `ATLAS_FP8_ROWWISE=1`, so
//! every other recipe's ledger is byte-identical to before this entry existed.
//!
//! **WHY (#917 H100 receipt, 2026-09-11, `Qwen/Qwen3.8-27B-FP8`).** The
//! `ATLAS_FP8_ROWWISE` GDN prefill arms dequantise their per-row FP8 weights
//! to BF16 once and multiply with cuBLASLt, because
//! `cublaslt::fp8_gemm_act_weight_t_rowwise` returns NOT_SUPPORTED on sm_121
//! (measured 2026-08-15). That dequant used to be a lazy `gpu.alloc` memoised
//! by weight pointer — `167772160` bytes for the fused `[QKV|Z]` weight PER
//! LAYER, invisible to `--gpu-memory-utilization`, which is the same defect
//! class that killed a 28-token prefill at layer 36 with `cuMemAlloc_v2
//! failed: status 2`. Sizing it here makes it ONE arena allocation the
//! preflight fitter can see (`preflight::headroom`'s `arena` term is
//! `BufferSizes::total_bytes()`), and the arms bump-carve their slices from
//! it instead of allocating.

use atlas_core::config::ModelConfig;

/// BF16 bytes ONE GDN layer's row-wise prefill arms dequantise and keep:
/// the fused `in_proj_qkvz` `[ssm_qkvz_size, hidden]` and `out_proj`
/// `[hidden, value_dim]`, 2 bytes per element.
///
/// EXACT, not an upper bound: these are the two weights
/// `set_fp8_rowwise_prefill_weights` installs and the only two the row-wise
/// arms dequantise. `value_dim = linear_num_value_heads * linear_value_head_dim`
/// — the same extent `trait_prefill_block.rs` passes as the `out_proj` K.
///
/// Qwen3.8-27B (hidden 5120, 16x128 key heads, 48x128 value heads):
/// `16384*5120*2 + 5120*6144*2 = 167772160 + 62914560 = 230686720` B.
pub fn ssm_rowwise_w_bf16_layer_bytes(config: &ModelConfig) -> usize {
    let bf16 = 2;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkvz = config.ssm_qkvz_size() * config.hidden_size * bf16;
    let out_proj = config.hidden_size * value_dim * bf16;
    qkvz + out_proj
}

/// Arena bytes for the row-wise GDN prefill BF16-weight slab: the per-layer
/// pair summed over every linear-attention layer, or 0 when the lever is off.
///
/// The env predicate is character for character
/// `weight_loader::qwen35_dense::rowwise_fp8::rowwise_fp8_enabled` — `== Ok("1")`,
/// NOT presence. The two must agree: the loader installs the per-row weights
/// on that predicate and the arms then require this slab, so `ATLAS_FP8_ROWWISE=0`
/// must leave BOTH off.
pub fn ssm_rowwise_w_bf16_bytes(config: &ModelConfig) -> usize {
    ssm_rowwise_w_bf16_bytes_for(
        config,
        std::env::var("ATLAS_FP8_ROWWISE").as_deref() == Ok("1"),
    )
}

/// [`ssm_rowwise_w_bf16_bytes`] with the lever passed in, so the sizing
/// arithmetic is pinnable at exact integers without touching the process
/// environment.
pub fn ssm_rowwise_w_bf16_bytes_for(config: &ModelConfig, rowwise_enabled: bool) -> usize {
    if !rowwise_enabled {
        return 0;
    }
    config.num_ssm_layers() * ssm_rowwise_w_bf16_layer_bytes(config)
}
