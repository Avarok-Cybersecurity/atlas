// SPDX-License-Identifier: AGPL-3.0-only

//! Shared kernel dispatch operations.
//!
//! Freestanding functions wrapping CUDA kernel launches via `KernelLaunch`.
//! Layer implementations compose these to build forward passes.
//!
//! Each function's parameters exactly match the corresponding CUDA kernel
//! signature. Grid/block dimensions are computed from the problem size.
//!
//! Refactor wave 4a (2026-05-03): split into `ops/` sub-modules with thematic
//! groupings. All public functions remain available at this path via re-export.

#[path = "ops/activations.rs"]
mod activations;
#[path = "ops/embeddings.rs"]
mod embeddings;
#[path = "ops/fp8_moe.rs"]
mod fp8_moe;
#[path = "ops/fp8_moe_batch_a.rs"]
mod fp8_moe_batch_a;
#[path = "ops/fp8_moe_batch_b.rs"]
mod fp8_moe_batch_b;
#[path = "ops/gemm_dense.rs"]
mod gemm_dense;
#[path = "ops/gemm_quant.rs"]
mod gemm_quant;
#[path = "ops/kv_cache.rs"]
mod kv_cache;
#[path = "ops/kv_cache_fp8k.rs"]
mod kv_cache_fp8k;
#[path = "ops/kv_cache_turbok.rs"]
mod kv_cache_turbok;
#[path = "ops/moe_expert.rs"]
mod moe_expert;
#[path = "ops/moe_expert_more.rs"]
mod moe_expert_more;
#[path = "ops/moe_gate.rs"]
mod moe_gate;
#[path = "ops/moe_grouped_a.rs"]
mod moe_grouped_a;
#[path = "ops/moe_grouped_b.rs"]
mod moe_grouped_b;
#[path = "ops/moe_prefill.rs"]
mod moe_prefill;
#[path = "ops/norm.rs"]
mod norm;
#[path = "ops/prefill_attn_a.rs"]
mod prefill_attn_a;
#[path = "ops/prefill_attn_b.rs"]
mod prefill_attn_b;
#[path = "ops/prefill_attn_batched.rs"]
mod prefill_attn_batched;
#[path = "ops/prefill_attn_fp8k.rs"]
mod prefill_attn_fp8k;
#[path = "ops/prefill_attn_main_a.rs"]
mod prefill_attn_main_a;
#[path = "ops/prefill_attn_main_b.rs"]
mod prefill_attn_main_b;
#[path = "ops/prefill_attn_turbok.rs"]
mod prefill_attn_turbok;
#[path = "ops/quant_dispatch.rs"]
mod quant_dispatch;
#[path = "ops/sampling.rs"]
mod sampling;
#[path = "ops/ssm_gdn_a.rs"]
mod ssm_gdn_a;
#[path = "ops/ssm_gdn_b.rs"]
mod ssm_gdn_b;
#[path = "ops/ssm_gdn_batched.rs"]
mod ssm_gdn_batched;
#[path = "ops/ssm_mamba.rs"]
mod ssm_mamba;
#[path = "ops/ssm_preproc.rs"]
mod ssm_preproc;

pub use activations::*;
pub use embeddings::*;
pub use fp8_moe::*;
pub use fp8_moe_batch_a::*;
pub use fp8_moe_batch_b::*;
pub use gemm_dense::*;
pub use gemm_quant::*;
pub use kv_cache::*;
pub use kv_cache_fp8k::*;
pub use kv_cache_turbok::*;
pub use moe_expert::*;
pub use moe_expert_more::*;
pub use moe_gate::*;
pub use moe_grouped_a::*;
#[allow(unused_imports)]
pub(crate) use moe_grouped_b::*;
pub use moe_prefill::*;
pub use norm::*;
pub use prefill_attn_a::*;
pub use prefill_attn_b::*;
pub use prefill_attn_batched::*;
pub use prefill_attn_fp8k::*;
pub use prefill_attn_main_a::*;
pub use prefill_attn_main_b::*;
pub use prefill_attn_turbok::*;
pub use quant_dispatch::*;
pub use sampling::*;
pub use ssm_gdn_a::*;
pub use ssm_gdn_b::*;
pub use ssm_gdn_batched::*;
pub use ssm_mamba::*;
pub use ssm_preproc::*;

/// Whether block-scaled FP8 prefill (per-128-block weight scales + per-token
/// activation scales via `fp8_gemm_t_blockscaled` / `moe_w8a8_grouped_gemm`)
/// is enabled. This is the DEFAULT for block-scaled FP8 checkpoints as of
/// 2026-06-17: it matches vLLM's per-block precision and avoids the
/// single-scale `fp8_gemm_n128` path, whose collapse of per-block dynamic
/// range pushed long-context tool-arg decode into the FP8 argmax-flip regime
/// (B1 drift gauge ~1400 → ~100 once block-scaled prefill is on).
///
/// Opt out with `ATLAS_FP8_SINGLE_SCALE=1` to restore the old single-scale
/// prefill (diagnostic / fallback only). Call sites still guard on the
/// presence of block-scaled weights + kernel handles, so builds/models
/// without those fall back automatically regardless of this flag.
pub fn fp8_blockscaled_prefill_enabled() -> bool {
    !matches!(
        std::env::var("ATLAS_FP8_SINGLE_SCALE").ok().as_deref(),
        Some("1")
    )
}
