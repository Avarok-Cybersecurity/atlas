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
mod op_dump;
mod prefill;
mod prefill_weights;
mod trait_impl;
mod types;

pub use types::{MlaWeights, Qwen3AttentionLayer};

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
