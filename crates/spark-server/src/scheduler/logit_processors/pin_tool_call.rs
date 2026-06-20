// SPDX-License-Identifier: AGPL-3.0-only

//! One-shot pin-to-`<tool_call>` immediately after `</think>`.
//!
//! Ported byte-for-byte from `decode_logits_seq::process_seq_logits`
//! lines ~247-261. When `seq.think_just_ended` is set (previous token
//! was `</think>`) AND the request requires a tool call AND no
//! tool-call has been opened yet AND we are no longer inside thinking,
//! blanket-mask all logits to `-inf` and set
//! `ctx.tool_call_start_token` to `0.0` so sampling deterministically
//! emits the structured tool-call opener.
//!
//! This prevents architectures like MiniMax M2 (which always thinks
//! via the chat template) from wandering into prose after `</think>`
//! instead of emitting the structured tool call. Requests without
//! `require_tool_call` (no tools passed) skip this entirely.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct PinToToolCallStart;

impl LogitsProcessor for PinToToolCallStart {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if a.think_just_ended
            && a.require_tool_call
            && !a.tool_call_opened
            && !a.inside_thinking
            && let Some(start_tok) = ctx.tool_call_start_token
        {
            let idx = start_tok as usize;
            if idx < logits.len() {
                for logit in logits.iter_mut() {
                    *logit = f32::NEG_INFINITY;
                }
                logits[idx] = 0.0;
                tracing::debug!(
                    "Forced tool_call_start_token after </think> (require_tool_call set)"
                );
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "pin_to_tool_call_start"
    }
}
