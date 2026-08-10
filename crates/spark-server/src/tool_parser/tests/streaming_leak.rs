// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

//! End-to-end streaming leak-suppression tests (split from
//! `streaming_frag.rs`, 500-LoC cap): detector -> `sanitize_content_chunk`
//! -> `flush_content_sanitizer` pipeline, spurious trailing-close
//! (Ornith `</_call>`) regressions, and `scrub_tool_tags` unit coverage.

use super::super::*;
use super::streaming_frag::write_and_bash_tools;

/// End-to-end streaming harness: feed each chunk through the detector
/// exactly as `handle_token` does, routing every `Content` payload
/// through the production `sanitize_content_chunk` with the qwen3_coder
/// leak markers. Returns `(tool_calls, content)` where `tool_calls` is
/// the list of `(name, args_json)` reassembled from the incremental
/// Start/Fragment/Delta/End (or bulk `ToolCall`) events, and `content`
/// is the concatenation of every post-sanitizer content chunk — i.e.
/// exactly what the client would see in `choices[].delta.content`.
fn stream_through_pipeline(chunks: &[&str]) -> (Vec<(String, String)>, String) {
    use crate::tool_parser::LeakMarkers;
    let markers: LeakMarkers = Qwen3CoderParser.leak_markers();
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());

    // Sanitizer state (mirrors the per-request fields on `StreamState`).
    let mut tag_scan_buf = String::new();
    let mut suppressing = false;
    let mut inside_env = false;

    // Tool-call accumulators keyed by idx (mirrors streaming_tool_args).
    let mut tc_acc: std::collections::BTreeMap<usize, (String, String)> =
        std::collections::BTreeMap::new();
    let mut content = String::new();

    let run = |out: &[DetectorOutput],
               tc_acc: &mut std::collections::BTreeMap<usize, (String, String)>,
               content: &mut String,
               tag_scan_buf: &mut String,
               suppressing: &mut bool,
               inside_env: &mut bool| {
        for o in out {
            match o {
                DetectorOutput::Content(text) => {
                    // Content → Tool boundary handling in production runs a
                    // `flush_content_sanitizer` on tool events; here the
                    // content arm just sanitizes (the leak we test is pure
                    // content, never a tool event).
                    let s = crate::api::sanitizer::sanitize_content_chunk(
                        text,
                        tag_scan_buf,
                        suppressing,
                        inside_env,
                        &markers,
                    );
                    content.push_str(&s);
                }
                DetectorOutput::ToolCallStart { name, idx, .. } => {
                    tc_acc.insert(*idx, (name.clone(), String::new()));
                }
                DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                    tc_acc.entry(*idx).or_default().1.push_str(fragment);
                }
                DetectorOutput::ToolCallDelta { args, idx } => {
                    tc_acc.entry(*idx).or_default().1.push_str(args);
                }
                DetectorOutput::ToolCall(tc, idx) => {
                    tc_acc.insert(
                        *idx,
                        (tc.function.name.clone(), tc.function.arguments.clone()),
                    );
                }
                DetectorOutput::ToolCallEnd { .. } => {}
            }
        }
    };

    for c in chunks {
        let out = det.process(c);
        run(
            &out,
            &mut tc_acc,
            &mut content,
            &mut tag_scan_buf,
            &mut suppressing,
            &mut inside_env,
        );
    }
    let out = det.flush();
    run(
        &out,
        &mut tc_acc,
        &mut content,
        &mut tag_scan_buf,
        &mut suppressing,
        &mut inside_env,
    );
    // Final sanitizer-tail flush (handle_done).
    content.push_str(&crate::api::stream_guards::flush_content_sanitizer(
        &mut tag_scan_buf,
        &mut suppressing,
        &markers,
    ));

    (tc_acc.into_values().collect(), content)
}

/// Regression: a model that appends a SPURIOUS redundant close marker
/// right after the real `</tool_call>` (Ornith-1.0 emits `</_call>` —
/// which BPE-tokenizes as `</` `_` `call` `>` — or a doubled
/// `</tool_call>`) must not leak that close into streamed `content`.
/// The real tool call must still parse with the correct name + args.
///
/// Live repro (2026-06-26): after 3 correct tool-call deltas the client
/// received a content delta of exactly `"\n\n</_call>"`. The trailing
/// close has no active orphan suppression (the detector already consumed
/// the real `</tool_call>`), so it must be dropped by the standalone
/// OrphanClose path in `sanitize_content_chunk`, including when split
/// across single-token chunks.
#[test]
fn qwen3_coder_spurious_trailing_close_underscore_call_not_leaked() {
    // One token per chunk for the 4-token `</_call>` tail, mirroring the
    // real BPE split (id-248059 `</tool_call>` is whole; `</_call>` is
    // `</` `_` `call` `>`).
    let chunks = [
        "<tool_call>",
        "<function=Bash>",
        "<parameter=command>",
        "ls -la",
        "</parameter>",
        "</function>",
        "</tool_call>",
        "\n\n", // whitespace the model emits before the spurious close
        "</",
        "_",
        "call",
        ">",
    ];
    let (calls, content) = stream_through_pipeline(&chunks);
    assert_eq!(calls.len(), 1, "exactly one tool call: {calls:?}");
    assert_eq!(calls[0].0, "Bash", "tool name preserved");
    let args: serde_json::Value = serde_json::from_str(&calls[0].1)
        .unwrap_or_else(|e| panic!("args not valid JSON: {e}; raw={:?}", calls[0].1));
    assert_eq!(args["command"], "ls -la", "args preserved");
    assert!(
        !content.contains("</_call>") && !content.contains("</tool_call>"),
        "spurious trailing close leaked into content: {content:?}"
    );
}

/// Companion case: a doubled real close `</tool_call></tool_call>`. The
/// second `</tool_call>` is a single vocab token, so it arrives as one
/// chunk — still must be dropped, not streamed as content.
#[test]
fn qwen3_coder_doubled_tool_call_close_not_leaked() {
    let chunks = [
        "<tool_call>",
        "<function=Bash>",
        "<parameter=command>",
        "echo hi",
        "</parameter>",
        "</function>",
        "</tool_call>",
        "</tool_call>", // spurious doubled close (whole token)
    ];
    let (calls, content) = stream_through_pipeline(&chunks);
    assert_eq!(calls.len(), 1, "exactly one tool call: {calls:?}");
    assert_eq!(calls[0].0, "Bash", "tool name preserved");
    let args: serde_json::Value = serde_json::from_str(&calls[0].1)
        .unwrap_or_else(|e| panic!("args not valid JSON: {e}; raw={:?}", calls[0].1));
    assert_eq!(args["command"], "echo hi", "args preserved");
    assert!(
        !content.contains("</tool_call>"),
        "spurious doubled close leaked into content: {content:?}"
    );
}

/// Direct guard on the buggy spot (`flush_content_sanitizer`). The live
/// leak (`content: '\n\n</_call>'`) reaches the client when the spurious
/// close is still sitting in the held-back tail buffer at stream end —
/// either fully assembled (`\n\n</_call>` / `\n\n</tool_call>`) or cut off
/// before its final `>` (`\n\n</_call`, when the closing token fused with
/// EOS). The pre-fix flush returned the tail VERBATIM because the leading
/// whitespace made `looks_like_partial_tag` (which requires the whole tail
/// to start with `<`) miss it. Assert every such tail flushes to just its
/// real-content prefix, with no close leak.
#[test]
fn flush_content_sanitizer_drops_held_back_trailing_close() {
    use crate::tool_parser::LeakMarkers;
    let markers: LeakMarkers = Qwen3CoderParser.leak_markers();
    let cases = [
        ("\n\n</_call>", "\n\n"),     // complete spurious close
        ("\n\n</tool_call>", "\n\n"), // complete doubled real close
        ("\n\n</_call", "\n\n"),      // incomplete: `>` never arrived
        ("ok</tool_call>", "ok"),     // close right after real content
        ("plain text", "plain text"), // no close → untouched
    ];
    for (tail, want) in cases {
        let mut buf = String::from(tail);
        let mut suppress = false;
        let out =
            crate::api::stream_guards::flush_content_sanitizer(&mut buf, &mut suppress, &markers);
        assert_eq!(
            out, want,
            "flush of {tail:?} should yield {want:?}, got {out:?}"
        );
        assert!(
            !out.contains("</_call") && !out.contains("</tool_call"),
            "close leaked from flush of {tail:?}: {out:?}"
        );
    }
}

/// End-to-end variant where the spurious `</_call>` reaches the FLUSH
/// (not the per-chunk sanitizer): the stream ends after `call` before the
/// closing `>` token arrives, so the streaming `sanitize_content_chunk`
/// holds the partial close in its tail and `flush_content_sanitizer` must
/// drop it. Without the fix this leaks `\n\n</_call` as a final content
/// chunk. The tool call still parses.
#[test]
fn qwen3_coder_spurious_trailing_close_reaches_flush_not_leaked() {
    let chunks = [
        "<tool_call>",
        "<function=Bash>",
        "<parameter=command>",
        "ls -la",
        "</parameter>",
        "</function>",
        "</tool_call>",
        "\n\n",
        "</",
        "_",
        "call", // stream ends here — the `>` token never came (fused w/ EOS)
    ];
    let (calls, content) = stream_through_pipeline(&chunks);
    assert_eq!(calls.len(), 1, "exactly one tool call: {calls:?}");
    assert_eq!(calls[0].0, "Bash", "tool name preserved");
    let args: serde_json::Value = serde_json::from_str(&calls[0].1)
        .unwrap_or_else(|e| panic!("args not valid JSON: {e}; raw={:?}", calls[0].1));
    assert_eq!(args["command"], "ls -la", "args preserved");
    assert!(
        !content.contains("</_call") && !content.contains("</tool_call"),
        "spurious trailing close leaked into content: {content:?}"
    );
}

/// Defense-in-depth: `scrub_tool_tags` must remove EVERY complete tool-call
/// markup tag from a content string, including the full raw block a desync'd
/// detector can dump on a runaway/truncation boundary. Uses the exact shape
/// reported live against Ornith-1.0-35B (leading spurious `</_call>`, multiple
/// `<tool_call><function=…><parameter=…>…</tool_call></_call>` blocks). Only
/// markup is removed; the inner argument *values* and surrounding whitespace
/// survive (they are not tool tags).
#[test]
fn scrub_tool_tags_strips_runaway_markup_dump() {
    use crate::tool_parser::LeakMarkers;
    let markers: LeakMarkers = Qwen3CoderParser.leak_markers();
    let dump = "</_call>\n<tool_call>\n<function=alarms>\n<parameter=category>\ngrid_power\n\
                </parameter>\n<parameter=window>\n24h\n</parameter>\n</function>\n</tool_call></_call>\n\
                <tool_call>\n<function=fleet>\n<parameter=limit>\n200\n</parameter>\n</function>\n\
                </tool_call></_call>";
    let scrubbed = crate::api::scrub::scrub_tool_tags(dump, &markers);
    for tag in [
        "<tool_call>",
        "</tool_call>",
        "</_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    ] {
        assert!(
            !scrubbed.contains(tag),
            "tag {tag:?} survived scrub: {scrubbed:?}"
        );
    }
    // Argument values are not tags and must remain.
    assert!(
        scrubbed.contains("grid_power"),
        "value dropped: {scrubbed:?}"
    );
    assert!(scrubbed.contains("200"), "value dropped: {scrubbed:?}");
}

/// `scrub_tool_tags` must leave legitimate prose untouched — a bare `<`
/// (math/comparison) and a non-tool `<...>` tag are not tool markup.
#[test]
fn scrub_tool_tags_preserves_non_tool_content() {
    use crate::tool_parser::LeakMarkers;
    let markers: LeakMarkers = Qwen3CoderParser.leak_markers();
    let prose = "if a < b and c > d, use <div> in the template";
    let scrubbed = crate::api::scrub::scrub_tool_tags(prose, &markers);
    assert_eq!(scrubbed, prose, "non-tool content altered");
}

/// End-to-end through the streaming pipeline: when the detector hands a full
/// raw tool-call block to the Content arm (the desync/runaway case — here
/// forced by a leading spurious `</_call>` that desyncs envelope detection),
/// no `<tool_call>`/`<function=`/`<parameter=`/`</_call>` markup may reach
/// the client's content stream.
#[test]
fn qwen3_coder_runaway_content_dump_not_leaked() {
    // Whole-string chunks (the detector classifies the leading stray close
    // and any unrecognised markup as Content, which routes through the
    // sanitizer state machine).
    let chunks = [
        "</_call>",
        "<tool_call><function=alarms><parameter=category>grid_power</parameter></function></tool_call></_call>",
    ];
    let (_calls, content) = stream_through_pipeline(&chunks);
    for tag in [
        "<tool_call>",
        "</tool_call>",
        "</_call>",
        "<function=",
        "<parameter=",
    ] {
        assert!(
            !content.contains(tag),
            "runaway markup {tag:?} leaked into content: {content:?}"
        );
    }
}
