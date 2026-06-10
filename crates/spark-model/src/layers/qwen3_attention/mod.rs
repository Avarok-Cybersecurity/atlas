// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3 full attention layer.
//!
//! Q/K/V projection -> Q/K norms -> RoPE -> KV cache write ->
//! paged decode attention -> O projection, then MoE FFN.
//!
//! Split into submodules:
//!   - `types`: `MlaWeights` + `Qwen3AttentionLayer` struct definitions
//!   - `init`: `new`, `new_ungated`, `new_with_gating` (kernel loading)
//!   - `helpers`: setters + `apply_layer_scalar` + `effective_attn_scale`
//!   - `prefill_weights`: prefill weight setup + W4A16 M128 dispatcher
//!   - `decode`: single-token attention forward + KV cache helpers
//!   - `prefill`: batched prefill with paged attention
//!   - `trait_impl`: `TransformerLayer` trait implementation

mod decode;
mod helpers;
mod init;
mod init_kernel_dispatch;
mod kernel_requirements;
mod op_dump;
// `innerq_driver` calls the CUDA Driver API directly via `atlas_core::registry`,
// which is itself gated on the `cuda` feature. Mirror that gate here so the
// metal-only build of spark-model (`--no-default-features --features metal`)
// compiles on Apple Silicon without dragging in `atlas_core::registry`.
#[cfg(feature = "cuda")]
pub mod innerq_driver;
mod prefill;
mod prefill_weights;
mod trait_impl;
mod types;

#[cfg(feature = "cuda")]
pub use innerq_driver::InnerQDriver;
pub use types::{MlaWeights, Qwen3AttentionLayer};

/// Startup fail-fast for `--kv-cache-dtype`: resolve every kernel handle the
/// dtype's dispatch arms require (chunked-prefill kernel, WHT bookends) and
/// error with the full missing list — BEFORE the multi-minute weight load,
/// instead of at first dispatch. See `kernel_requirements.rs`.
pub fn validate_required_kv_kernels(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    head_dim: usize,
) -> anyhow::Result<()> {
    kernel_requirements::validate_required_kernels(gpu, kv_dtype, head_dim)
}

#[cfg(feature = "cuda")]
use std::sync::OnceLock;

// Process-wide handle, populated at serve startup when `TURBO_INNERQ=N`
// is set. `OnceLock` matches Atlas's pattern for other singletons (kernel
// registry, EP comm). Kept here next to the driver itself so server code
// just does `qwen3_attention::INNERQ.get()`.
#[cfg(feature = "cuda")]
pub static INNERQ: OnceLock<InnerQDriver> = OnceLock::new();

/// Configured max decode batch size, set once at model init.
///
/// The split-K paged-attention split count is derived from this CONSTANT
/// rather than the runtime co-batched `num_seqs`. Previously
/// `num_splits = NUM_SMS / (num_q_heads * num_seqs)` made a sequence's
/// attention reduction tree depend on how many other sequences happened to be
/// co-batched in that step. The online-softmax split-merge is non-associative,
/// so the same sequence produced a few-ULP-different attention output (and a
/// different temp-0 argmax) when decoded alone vs co-batched — nondeterministic
/// output under concurrent load. Pinning the split count to the configured max
/// batch makes it invariant to co-batch count → deterministic.
/// See `tasks/determinism_investigation.md`.
static MAX_DECODE_SEQS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();

/// Record the configured max decode batch size (idempotent; first write wins).
/// Called once from `TransformerModel::new` with the serve `max_batch_size`.
pub fn set_max_decode_seqs(n: u32) {
    let _ = MAX_DECODE_SEQS.set(n.max(1));
}

/// Reference sequence count for the split-K split-count computation: the
/// configured max decode batch when set (the serve path always sets it), else
/// the runtime `num_seqs` (non-serve / test / graph-capture contexts). Clamped
/// to at least `num_seqs` so `num_splits` can never exceed what the fixed-size
/// split-K workspace (`NUM_SMS` slots) supports for the actual batch.
pub(crate) fn split_ref_seqs(num_seqs: u32) -> u32 {
    // NOTE (2026-06-03): tried unpinning this for num_seqs==1 to raise split-K
    // occupancy (16→48 CTAs) for single-stream long-ctx decode — clean A/B
    // (eqfix vs splitk, same 21.8k code task) was BYTE-IDENTICAL (12.7 tok/s
    // both), confirming attention occupancy is NOT the long-ctx bottleneck
    // (attention is ~5% of decode bytes at depth). Reverted. The real ~3.6x
    // decode gap vs vLLM is core kernel efficiency (MoE GEMV + per-step
    // overhead), a separate multi-week effort. Determinism pin kept intact.
    MAX_DECODE_SEQS
        .get()
        .copied()
        .unwrap_or(num_seqs)
        .max(num_seqs)
}
