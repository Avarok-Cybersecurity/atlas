// SPDX-License-Identifier: AGPL-3.0-only

//! Tool-body / parameter-body state machine (`update_tool_param_state`),
//! hoisted out of `emit_token` (SM1, 2026-05-26) and split into its own
//! file to keep `emit_step.rs` ≤500 LoC.

use super::super::types::ActiveSeq;

/// Tool-body / parameter-body state machine, hoisted out of
/// `emit_token` (SM1, 2026-05-26).
///
/// Both speculative-decoding paths (`verify_k2_step`, `verify_k4_step`,
/// `verify_dflash_step`, `spec_step`) and the regular non-spec decode
/// path (`decode_logits_step::process_decode_logits`) call this on
/// every emitted token so the state machine stays in sync with
/// `a.output_tokens`. The previous inline version was unreachable
/// from the non-spec path, leaving the close-tag mask, AM1 attractor
/// suppression, B1 margin detector, and A1 penalty toggle all silently
/// dead.
///
/// **Slice semantics**: this function does NOT assume `tok` has been
/// pushed onto `a.output_tokens` or that it has not. It auto-detects
/// from `a.output_tokens.last()` and slices accordingly:
///  - `emit_token` calls this BEFORE pushing → `last()` is the prior
///    token, lookback uses the full slice.
///  - `decode_logits_step::process_decode_logits` calls this AFTER
///    pushing → `last()` is `tok`, lookback excludes the trailing
///    entry so the search for `<parameter=KEY>` ending at the current
///    `>` is correct in both cases.
///
/// State mutations:
///  - `a.inside_tool_body`         set on `<tool_call>`, cleared on `</tool_call>`
///  - `a.tool_body_streak_tokens`  ++ per body token, reset on enter/exit
///  - `a.inside_parameter_body`    set on `<parameter=KEY>` close `>`, cleared on `</`
///  - `a.param_body_chars_emitted` ++ per non-close body token
///  - `a.finished`                 forced when stuck >MAX_TOOL_BODY_TOKENS
///
/// Token IDs are Qwen3.6 byte-level BPE (verified via /tokenize 2026-05-25):
///   27 = `<`, 28 = `=`, 29 = `>`, 510 = `</`, 15704 = `parameter`.
pub fn update_tool_param_state(a: &mut ActiveSeq, tok: u32) {
    const MAX_TOOL_BODY_TOKENS: u32 = 1024;
    if a.inside_thinking {
        return;
    }
    if a.tool_call_start_token == Some(tok) {
        a.inside_tool_body = true;
        a.tool_body_streak_tokens = 0;
        return;
    }
    if a.tool_call_end_token == Some(tok) {
        a.inside_tool_body = false;
        a.tool_body_streak_tokens = 0;
        a.inside_parameter_body = false;
        a.param_body_chars_emitted = 0;
        return;
    }
    if !a.inside_tool_body {
        return;
    }
    a.tool_body_streak_tokens = a.tool_body_streak_tokens.saturating_add(1);
    if a.tool_body_streak_tokens > MAX_TOOL_BODY_TOKENS {
        tracing::warn!(
            streak = a.tool_body_streak_tokens,
            "Stuck in tool body for {MAX_TOOL_BODY_TOKENS}+ tokens with no </tool_call>; ending response (model never closed the envelope — would otherwise burn to max_tokens). Sanitizer will salvage what it can."
        );
        a.finished = true;
    }

    const TOK_LT: u32 = 27;
    const TOK_PARAMETER: u32 = 15704;
    const TOK_EQ: u32 = 28;
    const TOK_GT: u32 = 29;
    const TOK_LT_SLASH: u32 = 510;

    if a.inside_parameter_body {
        if tok == TOK_LT_SLASH {
            // Start of `</parameter>` close-tag — exit body.
            a.inside_parameter_body = false;
            a.param_body_chars_emitted = 0;
        } else {
            // Any non-close body token advances the counter. The
            // position-0 mask in `decode_logits_seq.rs` (close-tag +
            // AM1 attractor) fires only while this counter is 0, so it
            // deactivates after the first emitted body token.
            a.param_body_chars_emitted = a.param_body_chars_emitted.saturating_add(1);
        }
        return;
    }

    // Not yet inside_parameter_body: scan for `<parameter=KEY>` opener
    // ending at this `>` (29). Lookback 8 tokens for `[27, 15704, 28]`
    // signature without an intervening close.
    if tok != TOK_GT {
        return;
    }
    // Auto-detect whether `tok` is already in output_tokens (caller
    // pushed) or not (caller has not yet pushed). The signature search
    // must NOT include `tok` itself — the lookback is "what came
    // BEFORE this `>`".
    let n = a.output_tokens.len();
    let n_for_lookback = if n > 0 && a.output_tokens[n - 1] == tok {
        n - 1
    } else {
        n
    };
    if n_for_lookback < 3 {
        return;
    }
    let start = n_for_lookback.saturating_sub(8);
    let window = &a.output_tokens[start..n_for_lookback];
    let mut sig_idx: Option<usize> = None;
    for i in 0..window.len().saturating_sub(2) {
        if window[i] == TOK_LT && window[i + 1] == TOK_PARAMETER && window[i + 2] == TOK_EQ {
            sig_idx = Some(i + 3);
        }
    }
    let Some(after_eq) = sig_idx else { return };
    // Check no intervening `</` or `>` in the KEY span between
    // `<parameter=` and the current `>`.
    let body_segment = &window[after_eq..];
    let intervening_close = body_segment
        .iter()
        .any(|&t| t == TOK_LT_SLASH || t == TOK_GT);
    if !intervening_close {
        a.inside_parameter_body = true;
        a.param_body_chars_emitted = 0;
    }
}

// SM1 unit tests deferred: ActiveSeq has 60+ fields and no public
// constructor; building a test instance requires more boilerplate
// than the state machine itself. Live-verification post-deploy is via
// the A1 rep-penalty toggle / B1 margin-detector behaviour.
