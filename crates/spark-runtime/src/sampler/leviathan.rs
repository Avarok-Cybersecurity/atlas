// SPDX-License-Identifier: AGPL-3.0-only
//
// Primitives for Leviathan-2023 speculative-decoding rejection
// sampling. Atlas's MTP proposer is argmax-only (no temperature, no
// penalties — see `spark-model/src/layers/mtp_head/forward.rs:382-488`),
// so the proposer's draft probability `p_draft(x) = 1.0` at the
// argmax token and 0 elsewhere. The general acceptance rule
// `min(1, p_target(x) / p_draft(x))` then collapses to `p_target(x)`
// and the residual on reject becomes `target` with the draft token's
// probability zeroed and renormalised. These two primitives are
// sufficient to implement that scheme:
//
// 1. `softmax_token_prob(logits, draft_id) -> p_target(draft_id)`
// 2. `sample_excluding(logits, params, history, draft_id) -> residual sample`
//
// The penalty-aware target distribution is constructed by the caller
// (`spark-server/src/scheduler/leviathan_verify.rs`) using the existing
// `apply_dry_penalty` / `apply_lz_penalty` pub helpers in the parent
// module; the verifier suppresses the OpenAI-style penalties
// (`repetition_penalty`, `presence_penalty`, `frequency_penalty`)
// because empirically (2026-05-11) they collapsed MTP acceptance to
// ≈0% by shifting argmax on every common code token. Only DRY+LZ
// fire on actual suffix-match repeats and are safe at the verify
// argmax position.

use super::{SamplingParams, sample_with_params_history};
use rand::{Rng, SeedableRng};

/// Softmax probability of `token_id` given raw FP32 logits.
///
/// Uses the max-subtract trick for numerical stability over large
/// vocabularies (Qwen3.6-27B vocab = 248320). Returns 0.0 when
/// `token_id` is out of bounds or when the post-shift sum-exp is
/// zero (all `-inf` logits). The output is in `[0.0, 1.0]`.
///
/// This is the per-token softmax used by Leviathan's acceptance
/// check; for a full distribution sample use
/// `sample_with_params_history` from the parent module.
pub fn softmax_token_prob(logits: &[f32], token_id: u32) -> f32 {
    let idx = token_id as usize;
    if idx >= logits.len() {
        return 0.0;
    }
    let mut max_logit = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_logit {
            max_logit = l;
        }
    }
    if !max_logit.is_finite() {
        return 0.0;
    }
    let mut sum_exp = 0.0_f32;
    for &l in logits {
        if l.is_finite() {
            sum_exp += (l - max_logit).exp();
        }
    }
    if sum_exp <= 0.0 {
        return 0.0;
    }
    let target = logits[idx];
    if !target.is_finite() {
        return 0.0;
    }
    ((target - max_logit).exp()) / sum_exp
}

/// Sample one token from `softmax(logits)` with `exclude_id`'s logit
/// forced to `-inf` before any further filtering. This implements the
/// Leviathan-2023 residual `max(0, p_target - δ_draft)` for the
/// argmax-proposer special case: the draft token is removed from the
/// candidate set, then we sample normally from the renormalised tail
/// using the caller's `params` and `history` (so DRY/LZ history-aware
/// penalties also apply to the residual sample).
///
/// The caller is responsible for applying any pre-softmax penalties
/// to `logits` first (DRY/LZ); this function only handles the
/// "exclude draft + sample" step.
pub fn sample_excluding(
    logits: &[f32],
    params: &SamplingParams,
    history: &[u32],
    exclude_id: u32,
) -> u32 {
    let n = logits.len();
    let mut buf = vec![0_u8; n * 4];
    for (i, &v) in logits.iter().enumerate() {
        let masked = if i == exclude_id as usize {
            f32::NEG_INFINITY
        } else {
            v
        };
        let bytes = masked.to_le_bytes();
        buf[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
    sample_with_params_history(&buf, params, history)
}

/// Deterministic uniform `[0.0, 1.0)` draw seeded from `seed`.
///
/// Used by the MTP rejection-sampling acceptance check
/// (`scheduler/leviathan_verify.rs`) so the verifier doesn't have to
/// pull `rand` as a direct dependency. Reproducible under fixed
/// `a.seed`: at decode step `n`, position `i`, the seed is derived
/// as `a.seed + output_tokens.len() + i` so successive steps draw
/// independent values without RNG state leaking across positions.
pub fn seeded_uniform_f32(seed: u64) -> f32 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    rng.r#gen::<f32>()
}
