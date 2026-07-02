// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

//! #192 parallel tool calls: a response containing SEVERAL
//! `<tool_call>…</tool_call>` blocks must surface ALL of them — the
//! non-stream parser as `tool_calls[0..n]` in emission order, the
//! streaming detector as per-call events with an incrementing index.
//! Covers hermes (JSON body) and qwen3_coder (XML body) fixtures, plus
//! single-call regressions pinning that the one-call shape is unchanged.

use super::super::*;

// ── helpers ─────────────────────────────────────────────────────────────

/// (kind, idx) trace of the indexed detector events, in emission order.
/// `kind` ∈ {"start", "end", "call"} — fragments/deltas are asserted
/// separately via [`args_for_idx`].
fn indexed_trace(outputs: &[DetectorOutput]) -> Vec<(&'static str, usize)> {
    outputs
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCallStart { idx, .. } => Some(("start", *idx)),
            DetectorOutput::ToolCallEnd { idx } => Some(("end", *idx)),
            DetectorOutput::ToolCall(_, idx) => Some(("call", *idx)),
            _ => None,
        })
        .collect()
}

/// Name announced by the `ToolCallStart` (or complete `ToolCall`) for `idx`.
fn name_for_idx(outputs: &[DetectorOutput], want: usize) -> Option<String> {
    outputs.iter().find_map(|o| match o {
        DetectorOutput::ToolCallStart { name, idx, .. } if *idx == want => Some(name.clone()),
        DetectorOutput::ToolCall(tc, idx) if *idx == want => Some(tc.function.name.clone()),
        _ => None,
    })
}

/// Reassembled arguments for `idx`: concatenated live fragments, else the
/// buffered `ToolCallDelta`, else the complete `ToolCall`'s arguments.
fn args_for_idx(outputs: &[DetectorOutput], want: usize) -> String {
    let frags: String = outputs
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCallArgsFragment { fragment, idx } if *idx == want => {
                Some(fragment.as_str())
            }
            _ => None,
        })
        .collect();
    if !frags.is_empty() {
        return frags;
    }
    outputs
        .iter()
        .find_map(|o| match o {
            DetectorOutput::ToolCallDelta { args, idx } if *idx == want => Some(args.clone()),
            DetectorOutput::ToolCall(tc, idx) if *idx == want => {
                Some(tc.function.arguments.clone())
            }
            _ => None,
        })
        .unwrap_or_default()
}

fn assert_json_eq(actual: &str, expected: &str, ctx: &str) {
    let a: serde_json::Value =
        serde_json::from_str(actual).unwrap_or_else(|e| panic!("{ctx}: bad JSON {actual:?}: {e}"));
    let b: serde_json::Value = serde_json::from_str(expected).unwrap();
    assert_eq!(a, b, "{ctx}: args mismatch");
}

/// Feed `text` to the detector in `chunk`-byte slices (ASCII fixtures), then
/// flush — exercises cross-chunk tag reassembly like the live token stream.
fn drive_chunked(det: &mut StreamingToolDetector, text: &str, chunk: usize) -> Vec<DetectorOutput> {
    let mut outputs = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let end = (i + chunk).min(text.len());
        outputs.extend(det.process(&text[i..end]));
        i = end;
    }
    outputs.extend(det.flush());
    outputs
}

fn hermes_call(name: &str, args: &str) -> String {
    format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>")
}

fn qwen3_coder_call(name: &str, params: &[(&str, &str)]) -> String {
    let mut s = format!("<tool_call>\n<function={name}>\n");
    for (k, v) in params {
        s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    s.push_str("</function>\n</tool_call>");
    s
}

// ── non-stream parser: all calls, in order ──────────────────────────────

#[test]
fn parse_hermes_two_calls_all_returned_in_order() {
    let text = format!(
        "{}\n{}",
        hermes_call("get_weather", r#"{"city": "Paris"}"#),
        hermes_call("get_time", r#"{"tz": "CET"}"#),
    );
    let (content, calls) = parse_tool_calls(&text);
    assert!(content.is_none(), "no content expected: {content:?}");
    assert_eq!(calls.len(), 2, "BOTH calls must be returned");
    assert_eq!(calls[0].function.name, "get_weather");
    assert_json_eq(
        &calls[0].function.arguments,
        r#"{"city":"Paris"}"#,
        "call 0",
    );
    assert_eq!(calls[1].function.name, "get_time");
    assert_json_eq(&calls[1].function.arguments, r#"{"tz":"CET"}"#, "call 1");
    assert_ne!(calls[0].id, calls[1].id, "each call gets a distinct id");
}

#[test]
fn parse_hermes_three_calls_same_name_distinct_args() {
    // BFCL `parallel` shape: the SAME function fanned out over 3 argument
    // sets in one response. Nothing may dedup or truncate at parse level.
    let text = ["Paris", "Berlin", "Tokyo"]
        .iter()
        .map(|c| hermes_call("get_weather", &format!(r#"{{"city": "{c}"}}"#)))
        .collect::<Vec<_>>()
        .join("\n");
    let (content, calls) = parse_tool_calls(&text);
    assert!(content.is_none());
    assert_eq!(calls.len(), 3, "all THREE same-name calls must survive");
    for (i, city) in ["Paris", "Berlin", "Tokyo"].iter().enumerate() {
        assert_eq!(calls[i].function.name, "get_weather");
        assert_json_eq(
            &calls[i].function.arguments,
            &format!(r#"{{"city":"{city}"}}"#),
            &format!("call {i}"),
        );
    }
}

#[test]
fn parse_qwen3_coder_two_calls_all_returned() {
    let text = format!(
        "{}\n{}",
        qwen3_coder_call("search", &[("query", "rust")]),
        qwen3_coder_call("read", &[("path", "/tmp/a.rs")]),
    );
    let (content, calls) = parse_tool_calls(&text);
    assert!(content.is_none(), "no content expected: {content:?}");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "search");
    assert_json_eq(
        &calls[0].function.arguments,
        r#"{"query":"rust"}"#,
        "call 0",
    );
    assert_eq!(calls[1].function.name, "read");
    assert_json_eq(
        &calls[1].function.arguments,
        r#"{"path":"/tmp/a.rs"}"#,
        "call 1",
    );
}

#[test]
fn parse_qwen3_coder_three_calls_with_content_around() {
    // Content BEFORE and AFTER the call block is legal — the client gets
    // content + tool_calls in the same message (OpenAI allows both).
    let text = format!(
        "Let me check all three.\n{}\n{}\n{}\nDone.",
        qwen3_coder_call("get_weather", &[("city", "Paris")]),
        qwen3_coder_call("get_weather", &[("city", "Berlin")]),
        qwen3_coder_call("get_weather", &[("city", "Tokyo")]),
    );
    let (content, calls) = parse_tool_calls(&text);
    assert_eq!(calls.len(), 3, "all three calls parsed");
    for (i, city) in ["Paris", "Berlin", "Tokyo"].iter().enumerate() {
        assert_eq!(calls[i].function.name, "get_weather");
        assert_json_eq(
            &calls[i].function.arguments,
            &format!(r#"{{"city":"{city}"}}"#),
            &format!("call {i}"),
        );
    }
    let content = content.expect("prose around the calls is preserved");
    assert!(content.contains("Let me check all three."));
    assert!(content.contains("Done."));
}

#[test]
fn parse_single_call_regression_unchanged() {
    // One call then EOS (EOS token itself is stripped upstream) — the
    // pre-#192 shape must stay byte-identical: one call, no content.
    let text = hermes_call("get_weather", r#"{"city": "Paris"}"#);
    let (content, calls) = parse_tool_calls(&text);
    assert!(content.is_none());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    assert_json_eq(
        &calls[0].function.arguments,
        r#"{"city":"Paris"}"#,
        "call 0",
    );
}

// ── streaming detector: incrementing index across calls ─────────────────

#[test]
fn streaming_hermes_two_calls_incrementing_index() {
    let text = format!(
        "{}\n{}",
        hermes_call("get_weather", r#"{"city": "Paris"}"#),
        hermes_call("get_time", r#"{"tz": "CET"}"#),
    );
    let mut det = StreamingToolDetector::new();
    let outputs = drive_chunked(&mut det, &text, 7);

    let trace = indexed_trace(&outputs);
    // Each call is announced under its own index, strictly increasing, and
    // call 0 fully closes before call 1 opens.
    let starts: Vec<usize> = trace
        .iter()
        .filter(|(k, _)| *k == "start" || *k == "call")
        .map(|(_, i)| *i)
        .collect();
    let ends: Vec<usize> = trace
        .iter()
        .filter(|(k, _)| *k == "end" || *k == "call")
        .map(|(_, i)| *i)
        .collect();
    assert_eq!(starts, vec![0, 1], "trace: {trace:?}");
    assert_eq!(ends, vec![0, 1], "trace: {trace:?}");
    assert_eq!(name_for_idx(&outputs, 0).as_deref(), Some("get_weather"));
    assert_eq!(name_for_idx(&outputs, 1).as_deref(), Some("get_time"));
    assert_json_eq(&args_for_idx(&outputs, 0), r#"{"city":"Paris"}"#, "idx 0");
    assert_json_eq(&args_for_idx(&outputs, 1), r#"{"tz":"CET"}"#, "idx 1");
}

#[test]
fn streaming_qwen3_coder_three_calls_incrementing_index() {
    let text = format!(
        "{}\n{}\n{}",
        qwen3_coder_call("get_weather", &[("city", "Paris")]),
        qwen3_coder_call("get_weather", &[("city", "Berlin")]),
        qwen3_coder_call("get_weather", &[("city", "Tokyo")]),
    );
    let mut det = StreamingToolDetector::new();
    let outputs = drive_chunked(&mut det, &text, 5);

    let trace = indexed_trace(&outputs);
    let starts: Vec<usize> = trace
        .iter()
        .filter(|(k, _)| *k == "start" || *k == "call")
        .map(|(_, i)| *i)
        .collect();
    assert_eq!(starts, vec![0, 1, 2], "trace: {trace:?}");
    for (i, city) in ["Paris", "Berlin", "Tokyo"].iter().enumerate() {
        assert_eq!(
            name_for_idx(&outputs, i).as_deref(),
            Some("get_weather"),
            "idx {i}"
        );
        assert_json_eq(
            &args_for_idx(&outputs, i),
            &format!(r#"{{"city":"{city}"}}"#),
            &format!("idx {i}"),
        );
    }
    assert!(det.has_tool_calls());
}

#[test]
fn streaming_single_call_close_streamed_regression() {
    // Post-#192 the scheduler streams `</tool_call>` and continues to EOS;
    // a single-call turn must still yield exactly ONE call at idx 0.
    let text = hermes_call("get_weather", r#"{"city": "Paris"}"#);
    let mut det = StreamingToolDetector::new();
    let outputs = drive_chunked(&mut det, &text, 6);
    let trace = indexed_trace(&outputs);
    assert!(
        trace.iter().all(|(_, i)| *i == 0),
        "single call stays at idx 0: {trace:?}"
    );
    assert_eq!(
        trace
            .iter()
            .filter(|(k, _)| *k == "end" || *k == "call")
            .count(),
        1,
        "exactly one completed call: {trace:?}"
    );
    assert_json_eq(&args_for_idx(&outputs, 0), r#"{"city":"Paris"}"#, "idx 0");
}

#[test]
fn streaming_single_call_dangling_close_flush_regression() {
    // Legacy shape: `</tool_call>` was a stop token and never reached the
    // stream — flush() must still recover the call (issue #33 contract).
    let mut det = StreamingToolDetector::new();
    let mut outputs =
        det.process("<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"tz\": \"CET\"}}\n");
    outputs.extend(det.flush());
    let trace = indexed_trace(&outputs);
    assert!(
        trace.iter().all(|(_, i)| *i == 0),
        "single dangling call stays at idx 0: {trace:?}"
    );
    assert_eq!(name_for_idx(&outputs, 0).as_deref(), Some("get_time"));
    assert_json_eq(&args_for_idx(&outputs, 0), r#"{"tz":"CET"}"#, "idx 0");
}

#[test]
fn streaming_two_calls_with_interleaved_content() {
    // Prose between calls streams as Content and does not disturb indices.
    let text = format!(
        "Checking Paris first. {} now Berlin: {} all done.",
        hermes_call("get_weather", r#"{"city": "Paris"}"#),
        hermes_call("get_weather", r#"{"city": "Berlin"}"#),
    );
    let mut det = StreamingToolDetector::new();
    let outputs = drive_chunked(&mut det, &text, 9);
    let starts: Vec<usize> = indexed_trace(&outputs)
        .iter()
        .filter(|(k, _)| *k == "start" || *k == "call")
        .map(|(_, i)| *i)
        .collect();
    assert_eq!(starts, vec![0, 1]);
    let content: String = outputs
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::Content(c) => Some(c.as_str()),
            _ => None,
        })
        .collect();
    assert!(content.contains("Checking Paris first."), "{content:?}");
    assert!(content.contains("all done."), "{content:?}");
}
