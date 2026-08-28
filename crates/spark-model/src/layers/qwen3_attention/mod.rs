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
// V4: `pub(crate)` so the DeepSeek-V4 weight loader (`weight_loader::deepseek_v4`)
// and the V4 attention submodules can call `helpers::yarn_rope_mscale`. Non-V4
// code paths are unaffected by the wider visibility.
pub(crate) mod helpers;
mod init;
mod init_arch_gates;
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
mod types_weights;

#[cfg(feature = "cuda")]
pub use innerq_driver::InnerQDriver;
// V4: re-export the new hyper-connection / compressor weight types alongside the
// existing ones. These are only constructed under DeepSeek-V4 detection.
pub(crate) use types::HeadGateActivation;
pub use types::Qwen3AttentionLayer;
pub use types_weights::{
    CompressorWeights, HcHeadWeights, HcLowRank, HcSiteWeights, HcWeights, MlaWeights,
};

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

// The InnerQ driver is owned by `TransformerModel` and reached through
// `Model::poll_innerq`. It used to live in a process-wide static here, which
// let it outlive the model whose `__device__` globals it writes.

/// Reference sequence count for the split-K split-count computation.
///
/// `num_splits = NUM_SMS / (num_q_heads * num_seqs)` made a sequence's
/// attention reduction tree depend on how many other sequences happened to be
/// co-batched in that step. The online-softmax split-merge is non-associative,
/// so the same sequence produced a few-ULP-different attention output (and a
/// different temp-0 argmax) when decoded alone vs co-batched — nondeterministic
/// output under concurrent load. Pinning the split count to the configured max
/// batch (`ModelLevers::max_decode_seqs`) makes it invariant to co-batch count.
/// See `tasks/determinism_investigation.md`.
///
/// Clamped to at least `num_seqs` so `num_splits` can never exceed what the
/// fixed-size split-K workspace (`NUM_SMS` slots) supports for the actual batch.
pub(crate) fn split_ref_seqs(num_seqs: u32, max_decode_seqs: u32) -> u32 {
    // NOTE (2026-06-03): tried unpinning this for num_seqs==1 to raise split-K
    // occupancy (16→48 CTAs) for single-stream long-ctx decode — clean A/B
    // (eqfix vs splitk, same 21.8k code task) was BYTE-IDENTICAL (12.7 tok/s
    // both), confirming attention occupancy is NOT the long-ctx bottleneck
    // (attention is ~5% of decode bytes at depth). Reverted. The real ~3.6x
    // decode gap vs vLLM is core kernel efficiency (MoE GEMV + per-step
    // overhead), a separate multi-week effort. Determinism pin kept intact.
    max_decode_seqs.max(num_seqs)
}

/// Host-time accumulator for the FFN/MoE half of prefill layers
/// (`ATLAS_PREFILL_HOST_TIMING=1`). Summed across layers and read+reset once
/// per prefill by the layer loop, so the attention half can be derived as
/// loop_wall - ffn.
pub static FFN_HOST_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn add_ffn_host_us(us: u64) {
    FFN_HOST_US.fetch_add(us, std::sync::atomic::Ordering::Relaxed);
}

pub fn take_ffn_host_us() -> u64 {
    FFN_HOST_US.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// Per-phase host-time accumulators for the prefill ATTENTION path
/// (`ATLAS_PREFILL_HOST_TIMING=1`). Index: 0=qkv projections, 1=everything
/// between qkv and the attention call (deinterleave + per-head norms + RoPE +
/// KV write), 2=the attention kernel call itself, 3=o_proj + head gate.
/// Summed across layers; read and reset once per prefill.
pub static ATTN_PHASE_US: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

pub fn add_attn_phase_us(i: usize, us: u64) {
    ATTN_PHASE_US[i].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
}

pub fn take_attn_phase_us() -> [u64; 4] {
    let mut o = [0u64; 4];
    for (i, a) in ATTN_PHASE_US.iter().enumerate() {
        o[i] = a.swap(0, std::sync::atomic::Ordering::Relaxed);
    }
    o
}

#[cfg(test)]
mod split_ref_seqs_tests {
    use super::split_ref_seqs;

    #[test]
    fn the_split_count_does_not_move_with_co_batch_size() {
        // The whole point of the pin: one sequence decoded alone and the same
        // sequence co-batched with fifteen others must see the same reduction
        // tree, or the non-associative split-merge flips its temp-0 argmax.
        let pin = 16;
        assert_eq!(split_ref_seqs(1, pin), split_ref_seqs(8, pin));
        assert_eq!(split_ref_seqs(1, pin), pin);
    }

    #[test]
    fn a_batch_larger_than_the_pin_clamps_up() {
        // `num_splits` must never exceed what the fixed-size split-K workspace
        // supports for the actual batch.
        assert_eq!(split_ref_seqs(32, 16), 32);
    }

    #[test]
    fn two_models_can_pin_to_different_batches() {
        // Was a `OnceLock`, so the second model to load silently kept the
        // first's max batch — and with it the first model's split count.
        assert_ne!(split_ref_seqs(1, 4), split_ref_seqs(1, 16));
    }
}
