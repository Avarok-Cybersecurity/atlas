// SPDX-License-Identifier: AGPL-3.0-only

//! Post-`</think>` symmetric mask for `</think>` and `<think>`.
//!
//! Ported byte-for-byte from `decode_logits_seq::process_seq_logits`
//! lines ~120-149. After thinking has ended (`seq.think_ended == true`),
//! mask BOTH the close token (`think_end_token` from ctx) and the
//! open token (`seq.think_start_token`) so the model cannot
//! degenerate into `</think></think>…` loops OR re-enter
//! `<think>` mid-response (F9, 2026-04-26).
//!
//! arXiv evidence (s1 / DeepSeek-R1 / Qwen3 / Production Repetition):
//! masking dominates penalty stacking for re-entry prevention.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct PostCloseThinkMask;

impl LogitsProcessor for PostCloseThinkMask {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if a.think_ended {
            if let Some(end_tok) = ctx.think_end_token {
                let end_idx = end_tok as usize;
                if end_idx < logits.len() {
                    logits[end_idx] = f32::NEG_INFINITY;
                }
            }
            if let Some(start_tok) = a.think_start_token {
                let start_idx = start_tok as usize;
                if start_idx < logits.len() {
                    logits[start_idx] = f32::NEG_INFINITY;
                }
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "post_close_think_mask"
    }
}
