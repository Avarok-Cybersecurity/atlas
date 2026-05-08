// SPDX-License-Identifier: AGPL-3.0-only

//! Regression tests for the qwen3_coder grammar's `required`-parameter
//! enforcement. Pinned by issue #40 (iromu, 2026-05-08) and the
//! OpenClaw multi-tool repro: Qwen3-Coder models constrained by the
//! qwen_xml_parameter grammar still emit `<tool_call>\n<function=NAME>\n
//! </function>\n</tool_call>` (zero `<parameter=>` blocks) even when
//! the JSON schema declares `required: [...]`.
//!
//! The bug lives upstream in `mlc-ai/xgrammar`'s
//! `cpp/json_schema_converter.cc::GetPartialRuleForProperties` — Case-1
//! (`min=0, max=-1`) emits `first_sep_rule | (property)*` when
//! `spec.min_properties == 0`, ignoring `spec.required`. The fix bumps
//! `min_properties` to `required.size()` when required is non-empty.
//!
//! These tests fail against `xgrammar v0.1.32` (the current pin) and
//! pass once the fork-with-fix is wired up via xgrammar-pins.toml.
//! See `.claude/plans/your-pr-for-issue-lexical-patterson.md`.

use super::*;
use xgrammar::{CompiledGrammar, GrammarMatcher};

fn exec_tool_def() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "exec".to_string(),
            description: Some("Run a shell command".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            })),
        },
    }]
}

/// Mirror of `tests/minimax.rs::grammar_accepts` — fresh matcher,
/// feed bytes, accept iff every byte parses AND the grammar reaches
/// an accepting (terminated) state.
fn grammar_accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher =
        GrammarMatcher::new(compiled, None, true, -1).expect("GrammarMatcher::new failed");
    if !matcher.accept_string(input, false) {
        return false;
    }
    matcher.is_terminated()
}

/// Positive baseline: the grammar must accept a properly-populated
/// `<tool_call>...<parameter=command>...</parameter>...</tool_call>`.
/// This pins the canonical happy path so a too-aggressive upstream fix
/// (e.g. one that breaks legitimate empty-string values) gets caught.
#[test]
fn qwen3_coder_grammar_accepts_canonical() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true)
        .expect("compile must succeed");

    let canonical =
        "<tool_call>\n<function=exec>\n<parameter=command>\nls /tmp\n</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, canonical),
        "canonical exec invocation must be accepted; input: {canonical:?}"
    );
}

/// Regression test for issue #40 / OpenClaw: when the schema declares
/// `required: ["command"]`, the grammar must REJECT a tool call body
/// with zero `<parameter=>` blocks. Failing this test means the
/// upstream xgrammar bug is still present.
///
/// Currently FAILS against xgrammar v0.1.32. Will PASS once the
/// `xgrammar-pins.toml` is updated to a fork with the
/// `min_properties >= required.size()` fix.
#[test]
fn qwen3_coder_grammar_rejects_empty_required_param() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true)
        .expect("compile must succeed");

    let empty_body = "<tool_call>\n<function=exec>\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, empty_body),
        "qwen3_coder grammar with required=['command'] must REJECT empty body \
         (no <parameter=> blocks). Currently accepts due to upstream xgrammar \
         bug in GetPartialRuleForProperties — Case-1 (min=0,max=-1) emits \
         `first_sep_rule | (property)*` when spec.min_properties==0, ignoring \
         spec.required. Fix: bump min_properties to required.size() in \
         json_schema_converter.cc. Input: {empty_body:?}"
    );
}
