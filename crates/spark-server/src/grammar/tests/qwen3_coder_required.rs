// SPDX-License-Identifier: AGPL-3.0-only

//! Regression tests for the qwen3_coder grammar's envelope shape.
//!
//! As of the body-format fix (2026-05-25): the grammar uses
//! `any_text` for the body inside `<tool_call>\n<function=NAME>\n…
//! \n</function>\n</tool_call>` to match the model's native XML
//! `<parameter=KEY>VALUE</parameter>` wire format. The body is
//! intentionally unconstrained at the grammar level — required-
//! parameter enforcement now happens host-side via
//! `validate_single_tool_call` and `backfill_required_params`
//! after `parse_one_call` extracts the XML/JSON args. See
//! `compile_tools.rs::compile_qwen3_coder_tool_grammar` and
//! `tool_handlers.rs:46` for the layered validation path.
//!
//! These tests pin the **envelope-shape** properties:
//! - Canonical bodies (XML or JSON) are ACCEPTED.
//! - Malformed envelopes (missing open/close tag) are REJECTED.
//!
//! The previous tests in this file asserted that the grammar
//! itself rejected empty-body tool calls — a property of the
//! prior `json_schema` body content type. Required-field
//! rejection is now the validator's responsibility, covered by
//! validator-side tests in `tool_parser/tests/`.

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

/// Positive baseline: the grammar must accept the canonical native
/// qwen3_coder XML body — the format the model is actually trained
/// to emit. Pins the wire-format envelope so a future grammar
/// rework cannot regress to forcing JSON-shape output (the exact
/// regression that caused interior-byte corruption and JSON
/// delimiter cascades against opencode multi-turn sessions in
/// 2026-05-25 sessions).
#[test]
fn qwen3_coder_grammar_accepts_canonical_xml_body() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let canonical_xml = "<tool_call>\n<function=exec>\n<parameter=command>ls /tmp</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, canonical_xml),
        "canonical native-XML qwen3_coder body must be accepted; input: {canonical_xml:?}"
    );
}

/// Regression test for issue #88 (Qwen3.6-27B-FP8, DGX Spark): when the
/// registered tools have shared name prefixes — the natural
/// `mcp_{server}_{tool}` MCP pattern where `mcp_scrapling_get` is a
/// textual prefix of `mcp_scrapling_get_prompt` — the per-tool auto-mode
/// triggers must stay prefix-free (closing `>` included) so xgrammar's
/// triggered-tags converter does NOT reject the whole grammar with "One
/// tag matches multiple triggers". Before the fix this hard-disabled
/// constrained decoding and the model concatenated/hallucinated tool
/// names (`skills_listskills_list`).
#[test]
fn qwen3_coder_grammar_compiles_with_shared_tool_name_prefixes() {
    fn mcp_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: crate::tool_parser::FunctionDefinition {
                name: name.to_string(),
                description: Some(format!("MCP tool {name}")),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "query": {"type": "string"} }
                })),
            },
        }
    }

    // `get` ⊂ `get_prompt` and `fetch` shares the `bulk_fetch` server
    // prefix — exactly the conflicting set reported in issue #88.
    let tools = vec![
        mcp_tool("mcp_scrapling_get"),
        mcp_tool("mcp_scrapling_get_prompt"),
        mcp_tool("mcp_scrapling_fetch"),
        mcp_tool("mcp_scrapling_bulk_fetch"),
    ];

    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    // AUTO mode (use_triggers=true) is the path that builds per-tool
    // triggers. Pre-fix this returned Err (both the qwen_xml_parameter
    // attempt and the json_schema fallback reuse the colliding triggers).
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("grammar with shared-prefix tool names must compile (issue #88)");

    // The longer, prefix-colliding tool must be reachable — not shadowed
    // by the shorter `mcp_scrapling_get` trigger.
    let long_call = "<tool_call>\n<function=mcp_scrapling_get_prompt>\n<parameter=query>\nhi\n</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, long_call),
        "longer prefix-colliding tool must be accepted; input: {long_call:?}"
    );

    // The shorter tool is still reachable too.
    let short_call = "<tool_call>\n<function=mcp_scrapling_get>\n<parameter=query>\nhi\n</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, short_call),
        "shorter tool must remain accepted; input: {short_call:?}"
    );
}

/// 2026-06-03: the qwen3_coder grammar intentionally enforces NATIVE XML
/// `<parameter=KEY>VALUE</parameter>` bodies and REJECTS a JSON-shaped body
/// (`<function=exec>{...}</function>`). The 2026-06-02 change
/// (compile_tools.rs:407) reverted an `any_text`/loose-body trial back to
/// the strict value EBNF *precisely because* the loose body let the model
/// freelance/ramble to finish=length — the runaway we are fighting.
/// Qwen3.6 emits native XML; MiniMax has its own grammar
/// (`compile_minimax_xml_tool_grammar`, also XML `<parameter name=>`, not
/// JSON). The JSON-shaped body remains supported by the PARSER fallback
/// (`parse_single_b.rs`) only when the grammar is disabled
/// (`--disable-tool-grammar`) — never by this grammar. (Was previously
/// `qwen3_coder_grammar_accepts_legacy_json_body`, an obsolete "supports
/// both shapes" contract superseded by the freelance fix.)
#[test]
fn qwen3_coder_grammar_rejects_json_body_enforces_xml() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let json_body =
        "<tool_call>\n<function=exec>\n{\"command\": \"ls /tmp\"}\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, json_body),
        "grammar must enforce native XML <parameter=>; the JSON body is parser-fallback \
         (grammar-off) only. input: {json_body:?}"
    );
}

/// Multi-parameter native-XML body — pins that consecutive
/// `<parameter=KEY>VALUE</parameter>` blocks are accepted without
/// the FSM clipping the closing `</parameter>` boundary (the exact
/// pruning failure that the 2026-05-23 sweep originally tried to
/// dodge by switching to JSON body).
#[test]
fn qwen3_coder_grammar_accepts_multi_xml_params() {
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "write".to_string(),
            description: Some("Write to a file".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filePath", "content"]
            })),
        },
    }];

    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let multi_param = "<tool_call>\n<function=write>\n<parameter=filePath>/tmp/test-rust-axum-v42/Cargo.toml</parameter>\n<parameter=content>[package]\nname = \"test-rust-axum-v42\"</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, multi_param),
        "multi-param native XML body must be accepted with full byte fidelity \
         (path tokens like `axum-v42` and content tokens with newlines/quotes). \
         Input: {multi_param:?}"
    );
}

/// Tier-0 non-empty enforcement (2026-05-25 evening): the qwen3_coder
/// grammar's regex content type must REJECT a parameter body that is
/// empty or whitespace-only. This is the Atlas-internal version of
/// llama.cpp#20164's "empty-parameter under long context" failure mode.
/// Without this, the model's in-tool sampler (which intentionally zeros
/// rep/DRY/freq/presence penalties because XGrammar usually shapes the
/// output) can pick `</parameter>` as its very next token after the
/// opener — burning opencode tool-call turns on empty bash commands and
/// empty file paths.
#[test]
fn qwen3_coder_grammar_rejects_empty_parameter_body() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let empty_body =
        "<tool_call>\n<function=exec>\n<parameter=command></parameter>\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, empty_body),
        "empty parameter body must be REJECTED by Tier-0 regex. Input: {empty_body:?}"
    );
}

#[test]
fn qwen3_coder_grammar_rejects_whitespace_only_parameter_body() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let whitespace_body = "<tool_call>\n<function=exec>\n<parameter=command>   \n  </parameter>\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, whitespace_body),
        "whitespace-only parameter body must be REJECTED. Input: {whitespace_body:?}"
    );
}

/// 2026-06-03: the content-start rule now ACCEPTS a leading newline before
/// real content (the model's genuine top-1 at the start of a write body).
/// The prior `first_char ::= [^ \t\r\n<]` masked `\n`, forcing the argmax
/// onto a drift runner-up (`lean`/`cargo`) under FP8 long-context. The
/// `leading_ws* first_content rest` rule permits the newline while still
/// requiring at least one real (non-ws) char (see the two reject tests).
#[test]
fn qwen3_coder_grammar_accepts_leading_newline_content() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let leading_nl = "<tool_call>\n<function=exec>\n<parameter=command>\nls /tmp</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, leading_nl),
        "a leading newline before real content must be ACCEPTED. Input: {leading_nl:?}"
    );
}

/// 2026-06-03 (diag agent acb6cb1): the param key closes with `>` and the
/// tokenizer fuses it with the value's first byte into a single `>X` merge
/// token (`>=`=id 9628). At the boundary the model can emit `>=`, depositing
/// a phantom `=` as the value's first char (`=axum::serve(...)`) — which
/// broke `edit` oldString matches and stalled the agent. `first_content`
/// now excludes `=`/`>`, so a value STARTING with `=` (the `>=`-merge
/// symptom) must be REJECTED by the grammar, forcing a clean `>`+content.
#[test]
fn qwen3_coder_grammar_rejects_eq_value_start() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let eq_start = "<tool_call>\n<function=exec>\n<parameter=command>=ls /tmp</parameter>\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, eq_start),
        "a value starting with `=` (the `>=`-merge artifact) must be REJECTED. Input: {eq_start:?}"
    );
}

/// 2026-07-09: the historical blanket `<`-ban on the value's first
/// content char made any `<`-initial file UNREPRESENTABLE — the mask
/// deflected `<script>`/`<!DOCTYPE` at position 0, so the model could not
/// write Svelte/HTML/XML files at all (live opencode session: every
/// `.svelte` write came out script-logic-only and the model retried the
/// identical write forever). `first_content` now reuses the negative-
/// prefix close ladder for its `<` arm: `<` is legal at content start
/// unless it begins the exact `</parameter>` close.
#[test]
fn qwen3_coder_grammar_accepts_lt_initial_value() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    // A Svelte component: starts with `<script>`, has an HTML template.
    let svelte = "<tool_call>\n<function=exec>\n<parameter=command><script>\n  let x = 1;\n</script>\n\n<main>\n  <button on:click={inc}>+</button>\n</main></parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, svelte),
        "a `<script>`-initial value (Svelte component) must be ACCEPTED"
    );

    // An HTML document: starts with `<!DOCTYPE html>`.
    let html = "<tool_call>\n<function=exec>\n<parameter=command><!DOCTYPE html>\n<html><body>hi</body></html></parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, html),
        "a `<!DOCTYPE`-initial value (HTML document) must be ACCEPTED"
    );

    // A `</x`-initial value (legal `</`-prefix that is NOT the close tag).
    let close_prefix = "<tool_call>\n<function=exec>\n<parameter=command></div> is a stray close tag</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, close_prefix),
        "a `</div`-initial value must be ACCEPTED (only the exact close is barred)"
    );
}

/// Companion to the `<`-unban: the non-empty guard MUST survive. An empty
/// body (the close tag as the first body token — the original Epoch-3
/// failure the blanket ban targeted) stays rejected because every
/// `first_content` alternative consumes at least one non-close byte.
#[test]
fn qwen3_coder_grammar_still_rejects_close_tag_as_first_body_token() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let empty_body =
        "<tool_call>\n<function=exec>\n<parameter=command></parameter>\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, empty_body),
        "an empty body (immediate close tag) must remain REJECTED"
    );
}
