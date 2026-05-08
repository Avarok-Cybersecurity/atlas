// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

// ────────────────────────────────────────────────────────────────────────
// Issue #33: streaming flush() must not re-emit ToolCallStart when the
// incremental path already emitted it.
//
// Reported by @gbanyan: SSE chunks contained two `tool_calls[0]` headers
// with two different `id` values, then args, breaking client dispatch.
// Root cause: when the model emits `<tool_call>...<function=NAME>...args`
// and the stream ends WITHOUT `</tool_call>` (closing tag is the stop
// token, not streamed), `flush()` returned `DetectorOutput::ToolCall`
// (a fresh complete call with a NEW id from `parse_one_call`) even though
// the streaming path had already emitted `DetectorOutput::ToolCallStart`
// with a different id. Downstream `handle_complete_tool_call` then sent
// a second `tool_call_start_chunk` to the SSE.
//
// Fix: in `flush()` when `current_tc_name.is_some()` (incremental start
// was already emitted), emit `ToolCallDelta + ToolCallEnd` matching the
// in-stream close path — not a full `ToolCall`.
// ────────────────────────────────────────────────────────────────────────

#[test]
fn flush_after_incremental_start_does_not_double_emit() {
    let mut det = StreamingToolDetector::new();

    // Simulate token-by-token streaming up to the function header.
    // After this, ToolCallStart has been emitted incrementally.
    let pre_flush =
        det.process("<tool_call>\n<function=web_search>\n<parameter=query>meeting</parameter>\n");
    let starts: Vec<_> = pre_flush
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallStart { .. }))
        .collect();
    assert_eq!(starts.len(), 1, "expected 1 incremental ToolCallStart");

    // Stream ends here — </tool_call> was the stop token, never streamed.
    // flush() must NOT emit a fresh `ToolCall` (which would carry a new id);
    // it should emit ToolCallDelta + ToolCallEnd against the existing header.
    let flushed = det.flush();
    let starts_after: Vec<_> = flushed
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallStart { .. }))
        .collect();
    let complete_after: Vec<_> = flushed
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCall(_, _)))
        .collect();
    assert!(
        starts_after.is_empty(),
        "flush() must not re-emit ToolCallStart when incremental start already happened"
    );
    assert!(
        complete_after.is_empty(),
        "flush() must not emit ToolCall (which carries a fresh id) — emit Delta+End instead"
    );

    // Must include the args delta + end so the client gets the complete call.
    let has_delta = flushed
        .iter()
        .any(|o| matches!(o, DetectorOutput::ToolCallDelta { .. }));
    let has_end = flushed
        .iter()
        .any(|o| matches!(o, DetectorOutput::ToolCallEnd { .. }));
    assert!(has_delta, "flush() must emit ToolCallDelta with the args");
    assert!(has_end, "flush() must emit ToolCallEnd to close the call");

    // The args delta should carry the parsed arguments (JSON), not empty.
    let args = flushed
        .iter()
        .find_map(|o| match o {
            DetectorOutput::ToolCallDelta { args, .. } => Some(args.clone()),
            _ => None,
        })
        .unwrap();
    assert!(
        args.contains("meeting"),
        "args should contain 'meeting'; got: {args}"
    );
}
