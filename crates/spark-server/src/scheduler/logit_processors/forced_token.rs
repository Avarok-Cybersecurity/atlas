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
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if !a.inside_thinking
            && a.top_logprobs.is_none()
            && ctx.sampling.forced_token_fastpath
            && let Some(ref mut gs) = a.grammar_state
            && let Some(forced) = gs.forced_token()
        {
            let forced = forced as u32;
            // `min_tokens` guard: the fast path bypasses sampling, so a
            // grammar-forced EOS would otherwise surface before the request
            // reached its minimum token count even though the normal path
            // masks EOS logits. Fall through to the standard pipeline (where
            // the mask applies) instead of emitting the EOS early.
            let effective_len = a.output_tokens.len().saturating_add(ctx.verify_pos);
            if effective_len < a.min_tokens && a.eos_tokens.contains(&forced) {
                return ProcessorOutcome::Continue;
            }
            return ProcessorOutcome::EmitToken(forced);
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "forced_token_fastpath"
    }
}
