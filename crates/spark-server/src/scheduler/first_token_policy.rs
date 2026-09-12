// SPDX-License-Identifier: AGPL-3.0-only

//! Token-0 grammar policy — the one seam every prefill route shares.
//!
//! Invariant: **a grammar constrains or advances token 0 iff the sequence
//! is born with `inside_thinking == false`.** While a sequence is inside
//! `<think>` the matcher is paused: no bitmask, no `accept_token`, and
//! `</think>` itself never advances it, so the first post-think token is
//! the first token the grammar ever sees. The decode loop already honours
//! this for tokens 1..N (`grammar_bitmask.rs`, `decode_logits_step.rs`,
//! `emit_step.rs` all gate on `!inside_thinking`). Token 0 did not: every
//! prefill route called `sample_first_token` with the armed grammar BEFORE
//! the `ActiveSeq` (and its `inside_thinking`) existed, so a
//! `required`/specific tool grammar forced `<tool_call>` into the reasoning
//! span at token 0, `JsonObject`/`JsonSchema` forced structural syntax
//! there, and the matcher was left one token ahead of the content stream
//! when `</think>` finally arrived (GLM-5.3 forced-tool request, 2026-09-06:
//! envelope fabricated inside `reasoning_content`, `tool_calls=null`).
//!
//! The fix is generic: the policy is derived from [`born_inside_thinking`],
//! the SAME predicate that births `ActiveSeq::inside_thinking`, so the
//! sampler cannot drift from the sequence it samples for. No model-,
//! grammar-class- or parser-specific branch exists here on purpose.

use crate::grammar::GrammarState;
use anyhow::Result;

/// The single birth predicate for `ActiveSeq::inside_thinking`.
///
/// Every `ActiveSeq` constructor and every [`FirstTokenPolicy`] derives
/// from this function. Do not re-derive thinking semantics anywhere else.
pub(super) fn born_inside_thinking(enable_thinking: bool, think_end_token: Option<u32>) -> bool {
    enable_thinking && think_end_token.is_some()
}

/// What token 0 may do with an armed grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FirstTokenPolicy {
    /// Token 0 is sampled inside `<think>`: skip the bitmask, skip
    /// `accept_token`, leave the matcher pristine for the first post-think
    /// token.
    pub grammar_suspended: bool,
    /// The `<tool_call>` id. Suppressed at token 0 while a grammar is armed
    /// and suspended — the decode-loop twin is `ToolCallDuringThinkingMask`,
    /// which only covers tokens 1..N.
    pub tool_call_start: Option<u32>,
}

impl FirstTokenPolicy {
    /// Policy for a sequence about to be born from these request facts.
    pub(super) fn for_birth(
        enable_thinking: bool,
        think_end_token: Option<u32>,
        tool_call_start: Option<u32>,
    ) -> Self {
        Self {
            grammar_suspended: born_inside_thinking(enable_thinking, think_end_token),
            tool_call_start,
        }
    }
}

/// Core of `sample_first_token`, generic over the model-dependent sampler.
///
/// `sample(suppress_ids, grammar)` is called exactly once. It receives the
/// grammar only when the policy lets the grammar act; when it does, the
/// matcher is advanced past the returned token (the decode loop's
/// `accept_token` only runs for tokens 1..N). With no grammar the call is
/// a plain pass-through — the non-grammar path is byte-identical to before.
pub(super) fn first_token_with<F>(
    policy: FirstTokenPolicy,
    suppress_ids: &[u32],
    grammar_state: Option<&mut GrammarState>,
    sample: F,
) -> Result<u32>
where
    F: FnOnce(&[u32], Option<&mut GrammarState>) -> Result<u32>,
{
    let Some(gs) = grammar_state else {
        return sample(suppress_ids, None);
    };
    if policy.grammar_suspended {
        // Grammar armed, sequence born inside <think>: the matcher must not
        // see this token. Keep the opener out of the reasoning span too.
        let mut ids = suppress_ids.to_vec();
        if let Some(t) = policy.tool_call_start
            && !ids.contains(&t)
        {
            ids.push(t);
        }
        return sample(&ids, None);
    }
    let tok = sample(suppress_ids, Some(&mut *gs))?;
    // A grammar-disallowed first token here would indicate the mask was not
    // applied — keep going rather than abort; the emit_step disengage path
    // handles any later desync gracefully.
    gs.accept_token(tok);
    Ok(tok)
}
