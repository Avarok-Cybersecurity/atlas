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
pub use types::Qwen3AttentionLayer;
pub use types_weights::{CompressorWeights, HcHeadWeights, HcSiteWeights, HcWeights, MlaWeights};

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

/// Split count for the split-K paged-attention decode path — the ONLY place it
/// may be computed.
///
/// Returns 1 when the single-pass kernel should be used instead.
///
/// Two properties, both load-bearing:
///
/// 1. **Determinism.** The count is derived from `split_ref_seqs` (the
///    configured max decode batch), not the runtime co-batched `num_seqs`, so a
///    sequence's non-associative online-softmax reduction tree is identical
///    alone vs co-batched. See `split_ref_seqs` above.
/// 2. **Memory safety.** The split-K workspace holds
///    `SPLITK_WORKSPACE_SLOTS` slots and the kernel writes one slot per
///    `(seq, q_head, split)`, so the dispatch MUST satisfy
///    `num_seqs * num_q_heads * num_splits <= SPLITK_WORKSPACE_SLOTS`.
///    Previously this bound held only *implicitly*, via the `.max(num_seqs)`
///    clamp inside `split_ref_seqs`, and the buffer-sizing comment stated the
///    bound with the `num_q_heads` factor missing. Anyone raising the split
///    count for occupancy — the obvious next optimization, since the grid is
///    only `num_q_heads * num_seqs` CTAs on `NUM_SMS` SMs — would have silently
///    written past the end of the workspace. It is now enforced here.
///
/// The clamp is a no-op for every split count the current heuristic produces
/// (`num_splits <= NUM_SMS / current_ctas` and `current_ctas >= num_q_heads *
/// num_seqs`, so the product is already `<= SPLITK_WORKSPACE_SLOTS`); it exists
/// so that a future change to the heuristic fails safe instead of corrupting
/// device memory. `split_k_bound_holds_for_all_reachable_shapes` proves both.
pub(crate) fn split_k_num_splits(num_q_heads: u32, num_seqs: u32) -> u32 {
    use atlas_core::device::sm121::{NUM_SMS, SPLITK_WORKSPACE_SLOTS};

    debug_assert!(num_q_heads >= 1 && num_seqs >= 1);
    let current_ctas = num_q_heads * split_ref_seqs(num_seqs);
    let num_splits = if current_ctas >= NUM_SMS {
        1
    } else {
        NUM_SMS / current_ctas
    };

    // Hard workspace bound. Never let the split count exceed what the
    // fixed-size workspace can hold for THIS dispatch.
    //
    // `.max(1)` on `max_splits` is deliberate and is NOT a bound violation: a
    // returned count of 1 means "use the single-pass kernel", which never
    // touches the workspace. So the invariant is conditional — it binds only
    // when the split-K path is actually taken.
    let slots_per_split = num_seqs.saturating_mul(num_q_heads).max(1);
    let max_splits = (SPLITK_WORKSPACE_SLOTS / slots_per_split).max(1);
    let num_splits = num_splits.min(max_splits);
    debug_assert!(
        num_splits == 1 || num_seqs * num_q_heads * num_splits <= SPLITK_WORKSPACE_SLOTS,
        "split-K workspace overflow: num_seqs={num_seqs} num_q_heads={num_q_heads} \
         num_splits={num_splits} needs {} slots > SPLITK_WORKSPACE_SLOTS={SPLITK_WORKSPACE_SLOTS}",
        num_seqs * num_q_heads * num_splits
    );
    num_splits
}

#[cfg(test)]
mod split_k_tests {
    use super::*;
    use atlas_core::device::sm121::{NUM_SMS, SPLITK_WORKSPACE_SLOTS};

    /// The workspace invariant must hold for every shape the decode path can
    /// present, and the bound must never silently shrink a split count that the
    /// existing heuristic already produced safely.
    #[test]
    fn split_k_bound_holds_for_all_reachable_shapes() {
        for num_q_heads in [1u32, 4, 8, 16, 24, 32, 40, 48, 64, 96] {
            for num_seqs in 1u32..=32 {
                let splits = split_k_num_splits(num_q_heads, num_seqs);
                assert!(splits >= 1, "split count must be at least 1");
                // The bound is CONDITIONAL: splits == 1 selects the single-pass
                // kernel, which never touches the workspace. (num_q_heads=4,
                // num_seqs=13 is exactly such a case: 52 > 48 slots, but
                // splits == 1 so nothing is written.)
                let slots = num_seqs * num_q_heads * splits;
                assert!(
                    splits == 1 || slots <= SPLITK_WORKSPACE_SLOTS,
                    "workspace overflow: nq={num_q_heads} ns={num_seqs} splits={splits} \
                     slots={slots} > {SPLITK_WORKSPACE_SLOTS}"
                );

                // Behaviour-preservation: identical to the legacy open-coded
                // formula, which was already within the bound.
                let current_ctas = num_q_heads * split_ref_seqs(num_seqs);
                let legacy = if current_ctas >= NUM_SMS {
                    1
                } else {
                    NUM_SMS / current_ctas
                };
                assert_eq!(
                    splits, legacy,
                    "clamp changed behaviour for nq={num_q_heads} ns={num_seqs}"
                );
            }
        }
    }

    /// The config this campaign measured: single-stream, no speculation, 24
    /// q-heads → 2 splits → 48 slots, i.e. the workspace is EXACTLY full.
    /// A regression that grew the split count here would run off the end.
    #[test]
    fn measured_config_fills_the_workspace_exactly() {
        let splits = split_k_num_splits(24, 1);
        assert_eq!(splits, 2);
        assert_eq!(1 * 24 * splits, SPLITK_WORKSPACE_SLOTS);
    }
}
