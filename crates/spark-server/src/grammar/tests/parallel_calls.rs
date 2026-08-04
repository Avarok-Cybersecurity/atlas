// SPDX-License-Identifier: AGPL-3.0-only

//! #192 generation-side multi-call property: with a trigger-based
//! (tool_choice="auto") hermes grammar, the matcher must accept a SECOND
//! `<tool_call>…</tool_call>` block after the first closes. This only holds
//! when the single-token `</tool_call>` (id 129 in the test vocab) actually
//! DRIVES the matcher — i.e. it is NOT in the `GrammarState` stop-token
//! exemption set. Before #192, `sampling_setup` pushed `</tool_call>` into
//! the request stop tokens for every tools-active request; those merged into
//! the eos set handed to `with_stop_tokens`, so `accept_token(129)`
//! short-circuited, the matcher never crossed the end-tag literal, and the
//! next call desynced the grammar (→ disengage → legacy one-call stop).

use super::*;

const TOOL_CALL_OPEN: u32 = 128; // "<tool_call>"
const TOOL_CALL_CLOSE: u32 = 129; // "</tool_call>"
const EOS: u32 = 130; // "<eos>"

/// Drive one complete hermes call through the matcher, byte tokens for the
/// JSON body and the single-token open/close tags — the exact shape a Qwen
/// tokenizer produces (`<tool_call>`/`</tool_call>` are single ids).
fn drive_one_call(state: &mut GrammarState, city: &str) {
    assert!(
        state.accept_token(TOOL_CALL_OPEN),
        "<tool_call> open must be accepted"
    );
    // Trigger tail + args object + hermes outer close, byte-wise. The tag's
    // begin literal is `<tool_call>\n{"name": "get_weather", "arguments": `
    // and its end literal is `}\n</tool_call>` (2026-07-03: tag format
    // mirrors the model's template-demonstrated shape); the args schema
    // fills the middle.
    let body =
        format!("\n{{\"name\": \"get_weather\", \"arguments\": {{\"location\":\"{city}\"}}}}\n");
    for (i, b) in body.bytes().enumerate() {
        assert!(
            state.accept_token(u32::from(b)),
            "body byte {i} ({:?}) must be grammar-legal",
            b as char,
        );
    }
    assert!(
        state.accept_token(TOOL_CALL_CLOSE),
        "single-token </tool_call> must ADVANCE the matcher (not exempt)"
    );
}

#[test]
fn hermes_auto_grammar_accepts_two_sequential_calls() {
    let vocab = test_vocab();
    let stop_ids = vec![EOS as i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    // auto mode: use_triggers=true (stop_after_first=false) — the grammar
    // must re-arm after each completed call.
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), true)
        .unwrap();
    // Production shape post-#192: ONLY real EOS ids are exempt —
    // `</tool_call>` is not, so it drives the matcher across the end tag.
    let mut state = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS]);

    drive_one_call(&mut state, "Paris");
    assert!(
        !state.is_terminated(),
        "auto mode: grammar must NOT terminate after the first call"
    );
    // The dispatch state between calls imposes no constraint — EOS is legal
    // (single-call turn) …
    assert!(
        state.stop_legal(&[EOS]),
        "EOS must be legal between completed calls"
    );
    // … and so is a SECOND call (parallel-call turn).
    drive_one_call(&mut state, "Berlin");
    assert!(
        state.stop_legal(&[EOS]),
        "EOS must be legal after the second call"
    );
}

/// Live failure #4 (2026-07-02 GB10 probe battery): tools armed, model makes
/// NO call ("just say hi") — the turn ran to `finish_reason="length"` because
/// the scheduler suppressed EOS via `!is_terminated()`, and an auto-mode
/// trigger grammar NEVER terminates. The correct predicate is positional
/// stop-legality: in the preamble/dispatch state EOS is grammar-legal, so
/// [`grammar_blocks_stop`] must NOT block the stop.
#[test]
fn armed_no_call_auto_grammar_never_blocks_eos() {
    let vocab = test_vocab();
    let stop_ids = vec![EOS as i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), true)
        .unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS]);

    // The exact condition the old scheduler gate keyed on — and why it was
    // wrong: not terminated, yet stopping is perfectly legal here.
    assert!(
        !state.is_terminated(),
        "auto grammar never terminates (this is WHY !is_terminated() was the wrong gate)"
    );
    assert!(
        !grammar_blocks_stop(Some(&mut state), &[EOS]),
        "armed-but-unused tools must not suppress EOS (preamble state permits end-of-turn)"
    );
    // Some prose, still no call — still free to stop.
    for b in b"hi there" {
        assert!(state.accept_token(u32::from(*b)));
    }
    assert!(!grammar_blocks_stop(Some(&mut state), &[EOS]));
    // No grammar at all (plain chat / disengaged) never blocks.
    assert!(!grammar_blocks_stop(None, &[EOS]));
}

/// EOS acceptance after the close literal: once a call completes (the matcher
/// ADVANCES across `</tool_call>` post-#192), the turn must be free to end —
/// whether or not another call follows. Mid-structure, the stop stays blocked.
#[test]
fn eos_reachable_after_each_close_literal_but_blocked_mid_call() {
    let vocab = test_vocab();
    let stop_ids = vec![EOS as i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), true)
        .unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS]);

    drive_one_call(&mut state, "Paris");
    assert!(
        !grammar_blocks_stop(Some(&mut state), &[EOS]),
        "a completed call must terminate cleanly (EOS legal after close literal)"
    );

    // Open a second call and stop INSIDE its JSON string value: end-of-turn
    // is grammar-illegal here, so the gate must block a (mask-leaked) EOS.
    assert!(state.accept_token(TOOL_CALL_OPEN));
    for b in b"\n{\"name\": \"get_weather\", \"arguments\": {\"location\":\"To" {
        assert!(state.accept_token(u32::from(*b)));
    }
    assert!(
        grammar_blocks_stop(Some(&mut state), &[EOS]),
        "EOS must stay suppressed mid-structure (open JSON string)"
    );

    // Close the second call — free to stop again.
    for b in b"kyo\"}}\n" {
        assert!(state.accept_token(u32::from(*b)));
    }
    assert!(state.accept_token(TOOL_CALL_CLOSE));
    assert!(
        !grammar_blocks_stop(Some(&mut state), &[EOS]),
        "EOS legal again after the SECOND close literal"
    );
}

/// tool_choice="required" (at_least_one + stop_after_first): the gate must
/// keep suppressing EOS until the mandatory call completes, then release.
#[test]
fn required_mode_blocks_eos_until_call_completes() {
    let vocab = test_vocab();
    let stop_ids = vec![EOS as i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), false)
        .unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS]);

    assert!(
        grammar_blocks_stop(Some(&mut state), &[EOS]),
        "required mode: EOS suppressed before the mandatory call"
    );
    drive_one_call(&mut state, "Paris");
    assert!(
        !grammar_blocks_stop(Some(&mut state), &[EOS]),
        "required mode: EOS legal once the mandatory call completed"
    );
}

/// Regression pin for the pre-#192 wedge: if `</tool_call>` is exempted as a
/// stop token, `accept_token` short-circuits WITHOUT advancing the matcher,
/// which is left mid-end-literal — a state that still constrains decoding
/// (the second `<tool_call>` cannot legally start). This is why the stop-token
/// push in `sampling_setup` had to go, not just the scheduler's legacy stop.
#[test]
fn hermes_grammar_wedges_if_tool_call_close_is_stop_exempt() {
    let vocab = test_vocab();
    let stop_ids = vec![EOS as i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), true)
        .unwrap();
    // Pre-#192 shape: `</tool_call>` merged into the exemption set.
    let mut state = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS, TOOL_CALL_CLOSE]);

    assert!(state.accept_token(TOOL_CALL_OPEN));
    let body = "\n{\"name\": \"get_weather\", \"arguments\": {\"location\":\"Paris\"}}\n";
    for b in body.bytes() {
        assert!(state.accept_token(u32::from(b)));
    }
    // Exempted: returns true but does NOT advance.
    assert!(state.accept_token(TOOL_CALL_CLOSE));
    // The matcher is stuck expecting the `</tool_call>` end literal: the
    // bitmask still constrains, and a second `<tool_call>` open is masked.
    assert!(
        state.fill_bitmask(),
        "wedged matcher still constrains decoding"
    );
    assert!(
        !state.is_token_allowed(TOOL_CALL_OPEN),
        "second <tool_call> is grammar-illegal while wedged mid-end-literal"
    );
}
