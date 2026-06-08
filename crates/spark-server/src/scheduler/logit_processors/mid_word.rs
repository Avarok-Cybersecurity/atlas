// SPDX-License-Identifier: AGPL-3.0-only

//! Mid-word `</think>` defer.
//!
//! Ported byte-for-byte from `decode_logits_seq::process_seq_logits`
//! lines ~107-118. While the model is INSIDE thinking, suppress the
//! `</think>` token whenever the previously emitted token decodes to
//! text ending in alphanumeric — i.e. the model is mid-word and
//! closing thinking now would yield "creating thep" / "ping/pong en"
//! style cuts (observed 2026-05-24 on Qwen3.6-FP8). The continuation
//! tokens (space/punctuation/newline) cap the defer at one extra step
//! most of the time.
//!
//! Fail-open: mask absent → no suppression.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;
use crate::scheduler::helpers::{disable_watchdogs, mid_word_token_mask};

pub struct MidWordThinkEndMask;

impl LogitsProcessor for MidWordThinkEndMask {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if !disable_watchdogs()
            && a.inside_thinking
            && let Some(end_tok) = ctx.think_end_token
            && let Some(prev_tok) = a.output_tokens.last().copied()
            && let Some(mask) = mid_word_token_mask()
            && mask.get(prev_tok as usize).copied().unwrap_or(false)
        {
            let end_idx = end_tok as usize;
            if end_idx < logits.len() {
                logits[end_idx] = f32::NEG_INFINITY;
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "mid_word_think_end_mask"
    }
}
