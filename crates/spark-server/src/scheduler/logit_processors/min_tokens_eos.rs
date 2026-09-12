// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-sampling EOS mask for the request `min_tokens` floor.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

/// Mask EOS ids while the effective output length is below `min_tokens`.
pub fn mask_eos_before_min(
    logits: &mut [f32],
    eos_tokens: &[u32],
    effective_len: usize,
    min_tokens: usize,
) {
    if min_tokens > 0 && effective_len < min_tokens {
        for &eos in eos_tokens {
            if let Some(slot) = logits.get_mut(eos as usize) {
                *slot = f32::NEG_INFINITY;
            }
        }
    }
}

/// Pipeline stage applying the mask to final decode and MTP verify positions.
pub struct MinTokensEosMask;

impl LogitsProcessor for MinTokensEosMask {
    fn apply(
        &self,
        logits: &mut [f32],
        seq: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        let effective_len = seq.output_tokens.len().saturating_add(ctx.verify_pos);
        mask_eos_before_min(logits, &seq.eos_tokens, effective_len, seq.min_tokens);
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "min_tokens_eos_mask"
    }
}

#[cfg(test)]
mod tests {
    use super::mask_eos_before_min;

    #[test]
    fn masks_only_before_boundary() {
        let mut logits = [0.5, 1.0, 2.0];
        mask_eos_before_min(&mut logits, &[1], 0, 5);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        let mut at_boundary = [0.5, 1.0, 2.0];
        mask_eos_before_min(&mut at_boundary, &[1], 5, 5);
        assert_eq!(at_boundary[1], 1.0);
    }

    #[test]
    fn zero_min_is_noop_and_oov_is_safe() {
        let mut logits = [0.5, 1.0];
        mask_eos_before_min(&mut logits, &[99_999], 0, 0);
        assert_eq!(logits, [0.5, 1.0]);
    }
}
