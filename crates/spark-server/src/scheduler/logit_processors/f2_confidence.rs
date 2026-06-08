// SPDX-License-Identifier: AGPL-3.0-only

//! F2 confidence-based early-stop arming.
//!
//! Ported byte-for-byte from `decode_logits_seq::process_seq_logits`
//! lines ~62-87. While the model is INSIDE `<think>…</think>` and has
//! emitted ≥ 400 thinking tokens, this stage tracks how many
//! consecutive tokens land at top-1 softmax probability ≥ 0.95. Once
//! the configured streak length is reached (see
//! [`crate::scheduler::confidence::confidence_run_step`]), it arms
//! `seq.force_end_thinking` so the downstream injector (stage 5) can
//! force `</think>` at a safe boundary.
//!
//! This stage NEVER modifies `logits` — it only reads them to compute
//! the top-1 prob and updates per-sequence accumulator state.
//! The decision to actually inject `</think>` lives in stage 5; this
//! stage's job is purely to arm the flag.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;
use crate::scheduler::confidence::confidence_run_step;

pub struct F2ConfidenceEarlyStop;

impl LogitsProcessor for F2ConfidenceEarlyStop {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        _ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if !crate::scheduler::helpers::disable_watchdogs()
            && a.inside_thinking
            && !a.force_end_thinking
            && a.thinking_tokens >= 400
            && crate::scheduler::helpers::watchdog_params().confidence_early_stop
        {
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
            let confident = sum_exp > 0.0 && 1.0 / sum_exp >= 0.95;
            let (run, force_end) = confidence_run_step(confident, a.consecutive_confident);
            a.consecutive_confident = run;
            if force_end {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                tracing::info!(
                    "Confidence early stop armed: top-1 prob >= 0.95 for {} tokens (after {} thinking tokens){}",
                    crate::scheduler::helpers::watchdog_params().confidence_run_length,
                    a.thinking_tokens,
                    if a.in_code_fence {
                        " — deferred until ``` fence closes"
                    } else {
                        " — deferring until next sentence boundary"
                    }
                );
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "f2_confidence_early_stop"
    }

    fn is_argmax_invariant(&self) -> bool {
        // Pure state-update stage: logits are never mutated.
        true
    }
}
