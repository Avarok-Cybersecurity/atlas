// SPDX-License-Identifier: AGPL-3.0-only

//! Forced-token fast-path (xgrammar Tier 3b, Coalescence).
//!
//! Ported byte-for-byte from `decode_logits_seq::process_seq_logits`
//! lines ~307-317. When the active tool-call grammar admits exactly
//! one legal next token, `forced_token()` returns `Some(id)` and the
//! model sample is redundant — the token is determined. Emitting `id`
//! directly is bit-identical to sampling from an all-but-`id`-masked
//! logit vector (every other token would be `-inf`).
//!
//! GUARDS — the fast-path fires only when ALL hold:
//!  * not inside `<think>` (thinking is unconstrained)
//!  * `top_logprobs` is NOT requested (logprobs need the distribution)
//!  * kill-switch enabled (`ATLAS_DISABLE_FORCED_TOKEN`)
//!  * an active grammar state exists
//!  * grammar reports exactly one legal next token
//!
//! This is the **only** stage that returns
//! [`ProcessorOutcome::EmitToken`].

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct ForcedTokenFastPath;

impl LogitsProcessor for ForcedTokenFastPath {
    fn apply(
        &self,
        _logits: &mut [f32],
        a: &mut ActiveSeq,
        _ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if !a.inside_thinking
            && a.top_logprobs.is_none()
            && crate::scheduler::helpers::forced_token_fastpath_enabled()
            && let Some(ref mut gs) = a.grammar_state
            && let Some(forced) = gs.forced_token()
        {
            return ProcessorOutcome::EmitToken(forced as u32);
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "forced_token_fastpath"
    }
}
