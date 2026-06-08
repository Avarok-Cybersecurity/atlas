// SPDX-License-Identifier: AGPL-3.0-only

//! Forced `</think>` injection (budget cap + confidence early-stop).
//!
//! Ported byte-for-byte from `decode_logits_seq::process_seq_logits`
//! lines ~175-236. Computes the three deferral inputs
//! (`at_sentence_boundary`, `defer_hard_override`,
//! `force_end_thinking`) and asks
//! [`crate::scheduler::confidence::should_inject_think_end`] whether
//! to inject `</think>` THIS step.
//!
//! When the gate fires, **blanket-mask all logits to `-inf` and set
//! `</think>` to `0.0`** so sampling is guaranteed to pick it. When
//! armed but deferring (force_end=true but gate=false), tick
//! `seq.sentence_defer_count` so the
//! [`crate::scheduler::confidence::MAX_SENTENCE_DEFER_TOKENS`] hard
//! ceiling stays bounded.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;
use crate::scheduler::confidence::{
    MAX_SENTENCE_DEFER_TOKENS, THINK_DEFER_ABS_CEILING, THINK_DEFER_BUDGET_FACTOR,
    should_inject_think_end,
};

pub struct ForcedThinkEndInjector;

impl LogitsProcessor for ForcedThinkEndInjector {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        let at_sentence_boundary = a
            .output_tokens
            .last()
            .copied()
            .and_then(|prev_tok| {
                crate::scheduler::helpers::boundary_token_mask()
                    .as_deref()
                    .and_then(|m| m.get(prev_tok as usize).copied())
            })
            .unwrap_or(false);
        let defer_hard_override = match a.thinking_budget {
            Some(b) => a.thinking_tokens >= b.saturating_mul(THINK_DEFER_BUDGET_FACTOR),
            None => a.thinking_tokens >= THINK_DEFER_ABS_CEILING,
        } || a.sentence_defer_count >= MAX_SENTENCE_DEFER_TOKENS;
        if a.inside_thinking
            && should_inject_think_end(
                a.force_end_thinking,
                a.in_code_fence,
                at_sentence_boundary,
                defer_hard_override,
            )
            && let Some(end_tok) = ctx.think_end_token
        {
            let end_idx = end_tok as usize;
            if end_idx < logits.len() {
                for logit in logits.iter_mut() {
                    *logit = f32::NEG_INFINITY;
                }
                logits[end_idx] = 0.0;
            }
        } else if a.inside_thinking && a.force_end_thinking {
            // Armed but deferring this step — tick the counter so the
            // MAX_SENTENCE_DEFER_TOKENS ceiling stays bounded.
            a.sentence_defer_count = a.sentence_defer_count.saturating_add(1);
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "forced_think_end_injector"
    }
}
