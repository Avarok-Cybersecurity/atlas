// SPDX-License-Identifier: AGPL-3.0-only

//! Width policy for the Qwen single-sequence verifier with live grammar state.

use super::super::ActiveSeq;

/// Only this tested model/configuration has acceptance-aware wide verification
/// and a proposer whose suffix is not sampled against a stale grammar mask.
/// Startup requires HC_BATCHED=1 whenever this Qwen proposer is active.
pub(in crate::scheduler) fn supported(seq: &ActiveSeq, dflash: bool) -> bool {
    let state = seq.seq.proposer_state.as_ref().and_then(|state| {
        state
            .as_any()
            .downcast_ref::<spark_model::layers::qwen4_exp_mtp_proposer::Qwen4ExpMtpProposerState>()
    });
    eligible(state.is_some(), dflash)
}

/// Caller bounds the requested width to one for concurrent grammar requests.
/// Batched and DFlash callers retain the original grammar clamp.
pub(in crate::scheduler) fn drafts(seq: &ActiveSeq, requested: usize, dflash: bool) -> usize {
    width(
        seq.grammar_state.is_some(),
        requested,
        supported(seq, dflash),
    )
}

fn eligible(qwen: bool, dflash: bool) -> bool {
    qwen && !dflash
}

fn width(has_grammar: bool, requested: usize, live: bool) -> usize {
    if live {
        requested.min(3)
    } else if has_grammar {
        1
    } else {
        requested
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_qwen_batched_highway_non_dflash_is_eligible() {
        for qwen in [false, true] {
            for dflash in [false, true] {
                assert_eq!(eligible(qwen, dflash), qwen && !dflash);
            }
        }
    }

    #[test]
    fn grammar_width_expands_only_for_supported_verifier_and_caps_at_three() {
        for requested in 1..=5 {
            assert_eq!(width(true, requested, true), requested.min(3));
            assert_eq!(width(true, requested, false), 1);
            assert_eq!(width(false, requested, false), requested);
        }
    }
}
