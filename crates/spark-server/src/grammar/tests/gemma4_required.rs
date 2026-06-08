// SPDX-License-Identifier: AGPL-3.0-only

//! Regression tests for the Gemma-4 tool grammar's `use_triggers` gating.
//!
//! 2026-06-06 fix: `compile_gemma4_tool_grammar` was the ONLY tool grammar
//! that pushed its trigger (`<|tool_call>call:`) UNCONDITIONALLY — even under
//! `tool_choice=required` (`use_triggers=false`). With a trigger present in
//! required mode, the structural-tag matcher is unconstrained until the
//! multi-byte trigger string completes token-by-token; on the GB10 BPE vocab
//! that trigger spans merges misaligned with the tag boundary and xgrammar's
//! token FSM can dead-end (HTTP 5xx) or emit a truncated tool name
//! ("read_fil"). The fix gates the trigger push on `use_triggers`, matching
//! the qwen3_coder grammar: in required mode the tag is enforced from token 1
//! via `at_least_one`/`stop_after_first`, with NO trigger window.
//!
//! These are byte-level contract tests (the live gemma-31b deploy exercises
//! the token-level path). They pin: (1) the canonical call is accepted in both
//! modes; (2) REQUIRED mode rejects leading prose (constrained from token 1 —
//! the fix's signature); (3) AUTO mode still allows free prose (trigger intact).

use super::*;
use xgrammar::{CompiledGrammar, GrammarMatcher};

fn read_file_tool() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "read_file".to_string(),
            description: Some("Read a file from disk".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": { "path": {"type": "string"} },
                "required": ["path"]
            })),
        },
    }]
}

/// Fresh matcher; accept iff every byte parses AND the grammar terminates.
fn grammar_accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher =
        GrammarMatcher::new(compiled, None, true, -1).expect("GrammarMatcher::new failed");
    if !matcher.accept_string(input, false) {
        return false;
    }
    matcher.is_terminated()
}

const CANONICAL: &str = "<|tool_call>call:read_file{\"path\": \"./Cargo.toml\"}<tool_call|>";

/// REQUIRED mode (use_triggers=false) must compile and accept the canonical
/// Gemma-4 framing. Pre-fix this path could compile to an FSM that dead-ends
/// at runtime; the canonical string must round-trip cleanly.
#[test]
fn gemma4_required_accepts_canonical_call() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_gemma4_tool_grammar(&read_file_tool(), false)
        .expect("required-mode gemma4 grammar must compile");
    assert!(
        grammar_accepts(&compiled, CANONICAL),
        "required-mode grammar must accept the canonical call; input: {CANONICAL:?}"
    );
}

/// AUTO mode (use_triggers=true) must still accept the canonical call — the
/// trigger gate must not break the auto path.
#[test]
fn gemma4_auto_accepts_canonical_call() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_gemma4_tool_grammar(&read_file_tool(), true)
        .expect("auto-mode gemma4 grammar must compile");
    assert!(
        grammar_accepts(&compiled, CANONICAL),
        "auto-mode grammar must accept the canonical call; input: {CANONICAL:?}"
    );
}

/// THE FIX SIGNATURE: in REQUIRED mode the grammar is constrained from token 1
/// (no trigger window), so a leading non-tag prose string must be REJECTED.
/// Before the fix a trigger was registered even in required mode, leaving the
/// pre-trigger window unconstrained.
#[test]
fn gemma4_required_rejects_leading_prose() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_gemma4_tool_grammar(&read_file_tool(), false)
        .expect("required-mode gemma4 grammar must compile");
    assert!(
        !grammar_accepts(&compiled, "Sure, let me read that file for you."),
        "required mode must REJECT leading prose (constrained from token 1)"
    );
}

/// AUTO mode legitimately allows free prose (the model chooses prose vs a tool
/// call) — the trigger gate must preserve this. Contrast with the required
/// rejection above; together they prove the gate flips correctly on use_triggers.
#[test]
fn gemma4_auto_allows_free_prose() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_gemma4_tool_grammar(&read_file_tool(), true)
        .expect("auto-mode gemma4 grammar must compile");
    assert!(
        grammar_accepts(&compiled, "Just a plain text answer, no tool needed."),
        "auto mode must ALLOW free prose (no forced tool call)"
    );
}
