// SPDX-License-Identifier: AGPL-3.0-only

//! Mistral `[TOOL_CALLS]` parsing — split from `group_b` for the file-size
//! cap. Self-contained: one parser family, no shared fixtures.

use super::super::*;

#[test]
fn parse_mistral_single_call() {
    let input = "[TOOL_CALLS]get_weather[ARGS]{\"location\":\"Paris\"}";
    let (c, calls) = parse_tool_calls(input);
    assert!(c.is_none());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Paris");
}

#[test]
fn parse_mistral_multiple_calls() {
    let input = "[TOOL_CALLS]search[ARGS]{\"q\":\"rust\"}[TOOL_CALLS]summarize[ARGS]{\"text\":\"found it\"}";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "search");
    assert_eq!(calls[1].function.name, "summarize");
    let a1: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
    assert_eq!(a1["text"], "found it");
}

#[test]
fn parse_mistral_with_leading_content() {
    let input = "Let me check.[TOOL_CALLS]get_weather[ARGS]{\"city\":\"Tokyo\"}";
    let (c, calls) = parse_tool_calls(input);
    assert_eq!(c.unwrap(), "Let me check.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
}

// MTP / speculative-decode fragmentation robustness tests live in
// the sibling `streaming_frag.rs` module to keep this file under the
// 500-LoC cap.
