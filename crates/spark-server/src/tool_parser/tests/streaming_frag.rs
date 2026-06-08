// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

//! MTP / speculative-decode fragmentation robustness for the
//! Qwen3-Coder streaming detector.
//!
//! vLLM PR #35615 ("Qwen3Coder streaming tool parser silently drops
//! parameters with speculative decoding") identified three bugs that
//! caused parameter loss when multi-token bursts arrived from spec
//! decode. Atlas's StreamingToolDetector is structurally immune
//! because it buffers everything between `<tool_call>` and
//! `</tool_call>` then parses the complete inner block — there is no
//! per-parameter early-return path that could drop fragments. These
//! tests lock that property in: deltas can be split at arbitrary byte
//! boundaries (mid-tag, mid-value, mid-XML opener) and the final
//! emitted arguments JSON must remain byte-exact.

use super::super::*;

#[test]
fn qwen3_coder_streaming_fragmented_at_path_boundary() {
    // Simulate MTP K=2 boundary splitting `/home/nologik` mid-string —
    // the failure shape from opencode-session.md where `/home/nologik`
    // arrived as `/home/nologin` (k → n char drop). If the buffer-
    // until-close design is correct, splitting the value mid-character
    // must not corrupt the final args.
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "<tool_call>",
        "<function=Read>",
        "<parameter=file_path>",
        "/home/nolo",  // first fragment ends mid-word
        "gik/test.rs", // second fragment completes path
        "</parameter>",
        "</function>",
        "</tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let mut args_json: Option<String> = None;
    for o in outputs {
        if let DetectorOutput::ToolCallDelta { args, .. } = o {
            args_json = Some(args);
        }
    }
    let args: serde_json::Value = serde_json::from_str(&args_json.expect("args emitted")).unwrap();
    assert_eq!(args["file_path"], "/home/nologik/test.rs");
}

#[test]
fn qwen3_coder_streaming_fragmented_at_xml_opener() {
    // Simulate spec-decode delivering a `<parameter=` opener split
    // across two deltas (`<param` then `eter=key>`). safe_emit_len
    // should hold back the partial tag instead of leaking it as
    // content; once complete it routes to the in-tag path.
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "<tool_call><function=Read>",
        "<param",          // partial tag
        "eter=file_path>", // tag completes
        "/etc/hosts</parameter></function></tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let mut args_json: Option<String> = None;
    for o in outputs {
        if let DetectorOutput::ToolCallDelta { args, .. } = o {
            args_json = Some(args);
        }
    }
    let args: serde_json::Value = serde_json::from_str(&args_json.expect("args emitted")).unwrap();
    assert_eq!(args["file_path"], "/etc/hosts");
}

#[test]
fn qwen3_coder_streaming_same_name_tool_calls_no_collision() {
    // vLLM bug 3 (name-based dedup in prev_tool_call_arr) would
    // collide two consecutive `Read` calls into one. Atlas keys by
    // call_counter, so two same-name calls must produce two distinct
    // outputs (whether ToolCall in bulk-fed mode or
    // ToolCallStart/Delta/End in incremental mode) with distinct
    // indices 0 and 1.
    //
    // This test uses bulk feed (close arrives in same chunk as
    // openers), which exercises the parse_one_call fast path and
    // emits two `ToolCall(tc, idx)` events.
    let mut det = StreamingToolDetector::new();
    let input = "<tool_call>\
                <function=Read>\
                <parameter=file_path>/a.rs</parameter>\
                </function>\
                </tool_call>\
                <tool_call>\
                <function=Read>\
                <parameter=file_path>/b.rs</parameter>\
                </function>\
                </tool_call>";
    let outputs = det.process(input);
    let calls: Vec<_> = outputs
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCall(tc, idx) => Some((
                *idx,
                tc.function.name.clone(),
                tc.function.arguments.clone(),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        calls.len(),
        2,
        "two ToolCall events for two same-name calls"
    );
    assert_eq!(calls[0].0, 0);
    assert_eq!(calls[1].0, 1);
    assert_eq!(calls[0].1, "Read");
    assert_eq!(calls[1].1, "Read");
    let args0: serde_json::Value = serde_json::from_str(&calls[0].2).unwrap();
    let args1: serde_json::Value = serde_json::from_str(&calls[1].2).unwrap();
    assert_eq!(args0["file_path"], "/a.rs");
    assert_eq!(args1["file_path"], "/b.rs");
}

#[test]
fn qwen3_coder_streaming_close_with_final_value_in_same_chunk() {
    // vLLM bug 1 (close-before-params ordering): a single burst
    // delivered `value</function>` together; their close check fired
    // first and dropped the value. Atlas's buffer-until-close design
    // means the value lands in the buffer BEFORE `</tool_call>` is
    // found; the close trigger then parses the whole inner block.
    // This test pins the property.
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "<tool_call><function=Write>",
        "<parameter=path>/tmp/x</parameter>",
        // Final param value and ALL closing tags arrive in one burst.
        "<parameter=content>hello world</parameter></function></tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let mut args_json: Option<String> = None;
    for o in outputs {
        if let DetectorOutput::ToolCallDelta { args, .. } = o {
            args_json = Some(args);
        }
    }
    let args: serde_json::Value = serde_json::from_str(&args_json.expect("args emitted")).unwrap();
    assert_eq!(args["path"], "/tmp/x");
    assert_eq!(
        args["content"], "hello world",
        "final-param-with-close burst must preserve the value"
    );
}
