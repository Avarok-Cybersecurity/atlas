// SPDX-License-Identifier: AGPL-3.0-only

//! SSOT for the on-GPU greedy fast-path penalty gate (#237, fix 4a).
//!
//! Both greedy-under-grammar fast paths — the MTP verify helper
//! (`verify_pipeline_helper::verify_pick_all_with_pipeline`) and the MTP
//! bootstrap sampler (`sample_step::sample_token_with_grammar`) — used to
//! require ALL penalties exactly neutral before skipping the host pipeline
//! (D2H + 248K-vocab dequant + mask + penalties + argmax). That gate is
//! stricter than necessary: at effective-greedy the emission is an argmax,
//! and **reduce-only** penalties provably cannot flip an argmax whose token
//! is immune.
//!
//! ## Provable bound
//!
//! Let `g` be the argmax of the RAW logits (the GPU argmax). The pipeline's
//! pick is the argmax of the masked-and-penalised logits. If:
//!  * every configured penalty only ever LOWERS logits of tokens in the
//!    scoped penalty history — `repetition_penalty >= 1.0` (divides positive
//!    logits: shrink; multiplies negative logits: more negative — both
//!    decreases), `presence_penalty >= 0.0` and `frequency_penalty >= 0.0`
//!    (subtract), while `lz_penalty == 0.0` and `dry_multiplier == 0.0`
//!    (those penalize pattern-EXTENDING tokens, not history members, so they
//!    fall outside the bound and must be off), and `logit_bias` is empty
//!    (bias can RAISE a competitor); and
//!  * `g` is NOT in the scoped history (its logit is untouched); and
//!  * `g`'s raw logit is > 0 (conservative guard, one 2/4-byte D2H);
//!
//! then after penalties every token's logit is `<=` its raw value while
//! `g`'s is unchanged, and `g` was already the raw maximum — so `g` remains
//! the argmax. Grammar legality of `g` is checked separately by the callers
//! (a grammar-allowed global max is the max of the allowed set).
//!
//! The scoped history used here MUST be the same one the pipeline hands to
//! `apply_penalties_and_bias` (`sample_step::penalty_history_scope`). A
//! `repetition_penalty_window` narrower than that history only shrinks the
//! penalized set, so membership in the full scoped history stays a
//! conservative superset test.

use spark_runtime::sampler::SamplingParams;

/// How the configured penalties interact with the greedy fast path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PenaltyGate {
    /// Every penalty is exactly neutral — the argmax is untouched, no
    /// per-token immunity check needed (the pre-#237 fast-path regime).
    Neutral,
    /// Reduce-only penalties (see module docs): fast path may fire per
    /// position iff [`argmax_immune`] holds for that position's argmax.
    ReduceOnly,
    /// Penalties/bias can raise a competitor or penalize non-history tokens
    /// (LZ/DRY) — the fast path must not fire.
    Blocked,
}

/// Classify the BUILT penalty params (post `penalty_params_for` gating) for
/// fast-path eligibility. `SamplingParams.logit_bias` must be empty for
/// anything but `Blocked`: bias entries are additive and can be positive.
pub(super) fn classify_penalties(p: &SamplingParams) -> PenaltyGate {
    if !p.logit_bias.is_empty() || p.lz_penalty != 0.0 || p.dry_multiplier != 0.0 {
        return PenaltyGate::Blocked;
    }
    if p.repetition_penalty == 1.0 && p.presence_penalty == 0.0 && p.frequency_penalty == 0.0 {
        return PenaltyGate::Neutral;
    }
    if p.repetition_penalty >= 1.0 && p.presence_penalty >= 0.0 && p.frequency_penalty >= 0.0 {
        return PenaltyGate::ReduceOnly;
    }
    PenaltyGate::Blocked
}

/// Per-position immunity check for [`PenaltyGate::ReduceOnly`]: the argmax
/// token is untouched by reduce-only penalties iff it is not in the scoped
/// history; `positive_logit` lazily reads the conservative raw-logit > 0
/// guard off the device (see [`logit_is_positive`]) — lazy because the
/// single-element D2H carries a stream sync, only paid after the membership
/// test passes.
pub(super) fn argmax_immune(
    tok: u32,
    scoped_history: &[u32],
    positive_logit: impl FnOnce() -> bool,
) -> bool {
    !scoped_history.contains(&tok) && positive_logit()
}

/// Read ONE logit for `tok` from the device logits buffer at `base`
/// (row `row` of a `[*, vocab]` layout) and test strict positivity.
/// BF16 unless the model reports the buffer as FP32. NaN and zero fail the
/// `> 0.0` comparison; any D2H error conservatively returns `false` (the
/// caller then falls back to the full host pipeline).
pub(super) fn logit_is_positive(
    model: &dyn spark_model::traits::Model,
    base: spark_runtime::gpu::DevicePtr,
    row: usize,
    vocab: usize,
    tok: u32,
) -> bool {
    let idx = row * vocab + tok as usize;
    let v = if model.logits_ptr_is_fp32(base) {
        let mut b = [0u8; 4];
        if model
            .copy_logits_to_host(base.offset(idx * 4), &mut b)
            .is_err()
        {
            return false;
        }
        f32::from_le_bytes(b)
    } else {
        let mut b = [0u8; 2];
        if model
            .copy_logits_to_host(base.offset(idx * 2), &mut b)
            .is_err()
        {
            return false;
        }
        crate::scheduler::helpers::bf16_to_f32(b[0], b[1])
    };
    v > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(rep: f32, pres: f32, freq: f32, lz: f32, dry: f32) -> SamplingParams {
        SamplingParams {
            // Sampling knobs irrelevant to the penalty gate; zeroed/neutral
            // explicitly because SamplingParams has no Default (PCND).
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            logit_bias: Vec::new(),
            repetition_penalty: rep,
            repetition_penalty_window: 0,
            presence_penalty: pres,
            frequency_penalty: freq,
            lz_penalty: lz,
            dry_multiplier: dry,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: None,
        }
    }

    #[test]
    fn neutral_params_classify_neutral() {
        assert_eq!(
            classify_penalties(&params(1.0, 0.0, 0.0, 0.0, 0.0)),
            PenaltyGate::Neutral
        );
    }

    #[test]
    fn reduce_only_params_classify_reduce_only() {
        assert_eq!(
            classify_penalties(&params(1.05, 0.0, 0.0, 0.0, 0.0)),
            PenaltyGate::ReduceOnly
        );
        assert_eq!(
            classify_penalties(&params(1.0, 0.5, 0.2, 0.0, 0.0)),
            PenaltyGate::ReduceOnly
        );
    }

    #[test]
    fn raising_or_pattern_penalties_block() {
        // rep < 1.0 RAISES history tokens' positive logits.
        assert_eq!(
            classify_penalties(&params(0.9, 0.0, 0.0, 0.0, 0.0)),
            PenaltyGate::Blocked
        );
        // Negative presence/frequency raise history tokens.
        assert_eq!(
            classify_penalties(&params(1.0, -0.1, 0.0, 0.0, 0.0)),
            PenaltyGate::Blocked
        );
        // LZ/DRY penalize pattern-extenders (not history members) — outside
        // the bound even at reduce-only signs.
        assert_eq!(
            classify_penalties(&params(1.0, 0.0, 0.0, 0.2, 0.0)),
            PenaltyGate::Blocked
        );
        assert_eq!(
            classify_penalties(&params(1.0, 0.0, 0.0, 0.0, 0.8)),
            PenaltyGate::Blocked
        );
    }

    #[test]
    fn logit_bias_blocks() {
        let mut p = params(1.0, 0.0, 0.0, 0.0, 0.0);
        p.logit_bias.push((42, -8.0));
        assert_eq!(classify_penalties(&p), PenaltyGate::Blocked);
    }

    #[test]
    fn immunity_requires_absence_and_positivity() {
        assert!(argmax_immune(7, &[1, 2, 3], || true));
        assert!(!argmax_immune(2, &[1, 2, 3], || true)); // in history
        assert!(!argmax_immune(7, &[1, 2, 3], || false)); // non-positive logit
    }

    #[test]
    fn positivity_read_is_lazy_after_membership_fail() {
        // The device read must NOT happen when the token is in history.
        let mut read = false;
        assert!(!argmax_immune(2, &[1, 2, 3], || {
            read = true;
            true
        }));
        assert!(!read);
    }
}
