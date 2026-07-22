// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

// ────────────────────────────────────────────────────────────────────────
// Echo-attractor salvage (2026-07-09). Regression fixtures from the live
// opencode collapse at 42.5k tokens on Qwen3.6-35B-A3B-FP8: the model
// emitted the XML wire format shifted by one structural token —
// `<parameter=parameter>filePath>\n/tmp/x</parameter>` and
// `<parameter=write>content>…</parameter>` — so the REAL key landed as the
// leading `IDENT>` of the value. `salvage_echoed_param` re-splits those,
// wired into `backfill_required_params` step 2.5 (buffered) and
// `streaming_emit::coerce_kv` (live fragments).
// ────────────────────────────────────────────────────────────────────────

fn write_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "write".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["content", "filePath"]
            })),
        },
    }
}

#[test]
fn salvage_helper_resplits_echoed_key() {
    let tools = vec![write_tool()];
    // Live shape 1: key slot echoed `parameter`, real key inside the value.
    let got = salvage_echoed_param(&tools, "write", "parameter", "filePath>\n/tmp/x/session.rs");
    assert_eq!(
        got,
        Some(("filePath".into(), "/tmp/x/session.rs".into())),
        "echoed `parameter` key with `filePath>` value prefix must re-split"
    );
    // Live shape 2: function name echoed into the key slot.
    let got = salvage_echoed_param(&tools, "write", "write", "content>use axum::Json;");
    assert_eq!(got, Some(("content".into(), "use axum::Json;".into())));
}

#[test]
fn salvage_helper_never_rewrites_legit_schema_keys() {
    let tools = vec![write_tool()];
    // Key IS a schema property: a value that happens to start with
    // `filePath>` is real content and must NOT be re-split.
    let got = salvage_echoed_param(&tools, "write", "content", "filePath> is the arg name");
    assert_eq!(got, None, "legit schema key must never be salvaged");
    // Unknown key but no schema-prop prefix in the value: no salvage.
    let got = salvage_echoed_param(&tools, "write", "bogus", "just some text");
    assert_eq!(got, None);
    // Unknown tool: no salvage.
    let got = salvage_echoed_param(&tools, "nosuch", "parameter", "filePath>/tmp/a");
    assert_eq!(got, None);
}

#[test]
fn buffered_pipeline_recovers_live_shape_1() {
    // Live failure shape 1: real content under its proper key, path echoed
    // under `parameter`. Before the fix this backfilled `filePath: ""` and
    // opencode threw BadResource, looping the degraded model forever.
    let input = "<tool_call>\n\
        <function=write>\n\
        <parameter=content>\nuse axum::Json;\n</parameter>\n\
        <parameter=parameter>filePath>\n/tmp/x/session.rs\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    let tool = write_tool();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["filePath"], "/tmp/x/session.rs",
        "path must be salvaged from the echoed key"
    );
    assert_eq!(args["content"], "use axum::Json;");
    assert!(
        args.get("parameter").is_none(),
        "echoed key must be consumed by the salvage, not left behind"
    );
}

#[test]
fn buffered_pipeline_recovers_live_shape_2() {
    // Live failure shape 2: BOTH keys echoed (function name + `parameter`),
    // both real keys leaked into the values.
    let input = "<tool_call>\n\
        <function=write>\n\
        <parameter=write>content>use sqlx::SqlitePool;\n</parameter>\n\
        <parameter=parameter>filePath>\n/tmp/x/session.rs\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    let tool = write_tool();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["content"], "use sqlx::SqlitePool;");
    assert_eq!(args["filePath"], "/tmp/x/session.rs");
    assert!(args.get("write").is_none());
    assert!(args.get("parameter").is_none());
}

#[test]
fn buffered_salvage_does_not_clobber_populated_target() {
    // If the real key was ALSO properly emitted (non-empty), the echoed
    // duplicate must not overwrite it.
    let input = "<tool_call>\n\
        <function=write>\n\
        <parameter=filePath>\n/tmp/real.rs\n</parameter>\n\
        <parameter=content>\nhello\n</parameter>\n\
        <parameter=parameter>filePath>\n/tmp/echoed.rs\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    let tool = write_tool();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["filePath"], "/tmp/real.rs",
        "populated key must win over the echo"
    );
}

#[test]
fn streaming_live_fragments_recover_echoed_key() {
    // Same live shape through the live-fragment streaming path (the actual
    // path of the failing session — opencode uses stream=true).
    let mut det = StreamingToolDetector::new_with_tools(vec![write_tool()]);
    let full = "<tool_call>\n<function=write>\n\
                <parameter=parameter>filePath>\n/tmp/x/session.rs\n</parameter>\n\
                <parameter=content>\nuse axum::Json;\n</parameter>\n\
                </function>\n</tool_call>";
    let mut outputs = Vec::new();
    for chunk in full.as_bytes().chunks(7) {
        outputs.extend(det.process(std::str::from_utf8(chunk).unwrap()));
    }
    outputs.extend(det.flush());
    let mut frags = String::new();
    for o in &outputs {
        if let DetectorOutput::ToolCallArgsFragment { fragment, .. } = o {
            frags.push_str(fragment);
        }
    }
    let args: serde_json::Value =
        serde_json::from_str(&frags).unwrap_or_else(|e| panic!("bad args json {frags:?}: {e}"));
    assert_eq!(args["filePath"], "/tmp/x/session.rs");
    assert_eq!(args["content"], "use axum::Json;");
    assert!(args.get("parameter").is_none());
}

// ── P0-2 (2026-07-09): garbled close-reopen (`</parameter<parameter=`) ──

#[test]
fn buffered_recovers_garbled_close_reopen() {
    // The 45k live shape: model dropped the `>` of a close. Buffered path
    // already re-splits at the reopen; the orphan `</parameter` tail must be
    // stripped from the first value.
    let input = "<tool_call>\n\
        <function=write>\n\
        <parameter=content>use axum::Json;\n</parameter<parameter=filePath>\n/tmp/x/+page.svelte\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    let tool = write_tool();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["content"], "use axum::Json;",
        "orphan </parameter tail must be stripped"
    );
    assert_eq!(args["filePath"], "/tmp/x/+page.svelte");
}

#[test]
fn streaming_recovers_garbled_close_reopen() {
    let mut det = StreamingToolDetector::new_with_tools(vec![write_tool()]);
    let full = "<tool_call>\n<function=write>\n\
                <parameter=content>use axum::Json;\n</parameter<parameter=filePath>\n/tmp/x/+page.svelte\n</parameter>\n\
                </function>\n</tool_call>";
    let mut outputs = Vec::new();
    for chunk in full.as_bytes().chunks(9) {
        outputs.extend(det.process(std::str::from_utf8(chunk).unwrap()));
    }
    outputs.extend(det.flush());
    let mut frags = String::new();
    for o in &outputs {
        if let DetectorOutput::ToolCallArgsFragment { fragment, .. } = o {
            frags.push_str(fragment);
        }
    }
    let args: serde_json::Value =
        serde_json::from_str(&frags).unwrap_or_else(|e| panic!("bad args json {frags:?}: {e}"));
    assert_eq!(
        args["filePath"], "/tmp/x/+page.svelte",
        "reopened param must be recovered live"
    );
    assert!(
        !args["content"].as_str().unwrap().contains("</parameter"),
        "garble must not leak into content"
    );
}

#[test]
fn legit_close_prefix_content_not_split() {
    // Negative: a value containing `</parameter` NOT followed by
    // `<parameter=` is real content (this file's own fixtures!) and must
    // survive intact through the buffered path via the proper close.
    let input = "<tool_call>\n\
        <function=write>\n\
        <parameter=filePath>/tmp/doc.md</parameter>\n\
        <parameter=content>the close tag is </parameter followed by text</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    let tool = write_tool();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["content"],
        "the close tag is </parameter followed by text"
    );
}
