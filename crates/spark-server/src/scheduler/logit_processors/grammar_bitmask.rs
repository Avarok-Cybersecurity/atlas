// SPDX-License-Identifier: AGPL-3.0-only

//! Grammar bitmask application.
//!
//! Ported byte-for-byte from `decode_logits_seq::process_seq_logits`
//! lines ~319-331. Applies the active grammar's next-token bitmask to
//! `logits` **before** sampling, but ONLY when outside thinking —
//! `<think>…</think>` is free-form reasoning stripped from the final
//! API response, so forcing it through a JSON-tool-call grammar
//! produces garbage punctuation streams (observed with opencode: the
//! assistant thinking channel filled with `!.,),,,***` before the
//! model recovered after `</think>`).

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct GrammarBitmaskApply;

impl LogitsProcessor for GrammarBitmaskApply {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        _ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if !a.inside_thinking
            && let Some(ref mut gs) = a.grammar_state
            && gs.fill_bitmask()
        {
            gs.apply_bitmask_to_logits(logits);
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "grammar_bitmask_apply"
    }
}
