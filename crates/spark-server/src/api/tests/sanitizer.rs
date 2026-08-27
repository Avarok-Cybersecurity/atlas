// SPDX-License-Identifier: AGPL-3.0-only

//! Content-sanitizer tests: orphan tool-call fragments the streaming
//! detector could not claim must never reach the client, and legitimate
//! prose must survive them.
//!
//! Assertions are on the WHOLE stream (chunks + end-of-stream flush); see
//! `super::harness`.

use super::harness::Stream;
use crate::tool_parser::{LeakMarkers, Qwen3CoderParser, ToolCallParser};

use crate::api::sanitizer::sanitize_content_chunk;
use crate::api::stream_guards::flush_content_sanitizer;

#[path = "sanitizer_chunk_tests.rs"]
mod sanitizer_tests;

fn qwen() -> LeakMarkers {
    Qwen3CoderParser.leak_markers()
}

#[test]
fn empty_markers_pass_text_through_untouched() {
    // A parser that opts out (Hermes, Gemma4, Mistral, BareJson) takes
    // the fast path: no buffering, so no added latency and no chance of
    // eating text it does not understand.
    let markers = LeakMarkers::EMPTY;
    let mut s = Stream::new(&markers);
    let first = s.feed("<parameter=foo>value</parameter>");
    assert_eq!(first, "<parameter=foo>value</parameter>");
    assert!(s.buffered().is_empty(), "no markers -> no tail buffering");
    assert!(!s.suppressing());
    assert_eq!(s.finish(), "<parameter=foo>value</parameter>");
}

#[test]
fn orphan_parameter_block_is_dropped_and_prose_survives() {
    // `<parameter=...>` outside a `<tool_call>` envelope is a half-formed
    // tool call the detector rejected. Suppress from the opener to the
    // first close tag; the stray `</function>` after it is dropped too.
    let markers = qwen();
    let mut s = Stream::new(&markers);
    s.feed("prefix<parameter=filePath>/tmp/x.txt</parameter>suffix</function>tail");
    assert_eq!(s.finish(), "prefixsuffixtail");
}

#[test]
fn a_tag_split_across_chunks_still_matches() {
    // The whole point of the tail buffer. `<param` + `eter=x>` must fuse
    // into one opener; if it did not, the leak would stream out verbatim.
    let markers = qwen();
    let mut s = Stream::new(&markers);
    let first = s.feed("abc<param");
    // Marker-prefix holdback (first-delta latency fix): the prose emits
    // immediately; only the `<param` suffix is withheld for fusion.
    assert_eq!(
        first, "abc",
        "prose before a possible tag prefix emits immediately"
    );
    assert!(!s.suppressing(), "a partial tag is not yet a leak");
    s.feed("eter=x>body</parameter>tail");
    assert_eq!(s.finish(), "abctail");
}

#[test]
fn a_leak_split_across_many_tiny_chunks_never_reaches_the_client() {
    // SSE deltas are token-sized, so every marker arrives fragmented in
    // production. Drive the same text one byte at a time.
    let markers = qwen();
    let mut s = Stream::new(&markers);
    s.feed_chunked("before<function=Bash>rm -rf /</function>after", 1);
    assert_eq!(s.finish(), "beforeafter");
}

#[test]
fn suppression_survives_a_chunk_boundary_inside_the_leak_body() {
    // The close tag arrives in a later chunk than the opener. Everything
    // between them is leak, however it is sliced.
    let markers = qwen();
    let mut s = Stream::new(&markers);
    s.feed("ok <tool_use>{\"name\":");
    assert!(s.suppressing(), "opener engages suppression");
    s.feed("\"x\"}");
    assert!(s.suppressing(), "still inside the leak");
    s.feed("</tool_use> done");
    assert!(!s.suppressing(), "close tag ends suppression");
    let out = s.finish();
    assert_eq!(out, "ok  done");
    assert!(!out.contains("name"), "leak body must not survive: {out:?}");
}

#[test]
fn legitimate_rust_prose_is_not_mistaken_for_a_tool_call() {
    // Real source says `fn add(...)`, never `<function=add>`. The angle
    // bracket is what makes the marker structural — prose about
    // functions, generics, and comparisons must pass through intact.
    let markers = qwen();
    let mut s = Stream::new(&markers);
    let prose = "Use `fn add(a: i32, b: i32) -> i32 { a + b }`; note `a < b` and `Vec<String>`.";
    s.feed(prose);
    assert_eq!(s.finish(), prose);
    // Same text, one byte at a time — chunking must not manufacture a
    // false positive out of a `<` that never completes a marker.
    let mut s = Stream::new(&markers);
    s.feed_chunked(prose, 1);
    assert_eq!(s.finish(), prose);
}

#[test]
fn flush_emits_a_pending_tail_when_no_markers_are_configured() {
    // With EMPTY markers nothing is ever buffered, but the flush must
    // still hand back whatever it holds rather than swallowing it.
    let markers = LeakMarkers::EMPTY;
    let mut buf = String::from("anything");
    let mut suppress = false;
    let out = crate::api::stream_guards::flush_content_sanitizer(&mut buf, &mut suppress, &markers);
    assert_eq!(out, "anything");
    assert!(buf.is_empty());
}

#[test]
fn flush_drops_a_dangling_partial_tag() {
    // The stream ended mid-marker. `<par` could only ever have become
    // `<parameter=`; emitting it would show the client the first bytes of
    // a leak. Dropping four characters is the cheaper error.
    let markers = qwen();
    let mut buf = String::from("<par");
    let mut suppress = false;
    let out = crate::api::stream_guards::flush_content_sanitizer(&mut buf, &mut suppress, &markers);
    assert_eq!(out, "");
}

#[test]
fn flush_clears_stuck_suppression_at_a_tool_boundary() {
    // The production bug this pins: the model emits `<parameter=` in
    // prose (suppression engages), then a REAL structured tool call
    // arrives and the detector consumes its `</parameter>` before the
    // sanitizer sees it. Without the flush at the Content -> Tool
    // boundary, `suppressing` stays true forever and eats the rest of
    // the response.
    let markers = qwen();
    let mut buf = String::new();
    let mut suppress = false;
    let mut env = false;

    let prose = crate::api::sanitizer::sanitize_content_chunk(
        "Let me write it: <parameter=content>foo",
        &mut buf,
        &mut suppress,
        &mut env,
        &markers,
    );
    assert_eq!(prose, "Let me write it: ");
    assert!(suppress, "orphan `<parameter=` enters suppression");

    let pre_tool =
        crate::api::stream_guards::flush_content_sanitizer(&mut buf, &mut suppress, &markers);
    assert_eq!(pre_tool, "", "the suppressed tail is dropped, not emitted");
    assert!(!suppress, "flush clears the suppression flag");
    assert!(buf.is_empty(), "flush clears the tail buffer");

    let mut s = Stream::new(&markers);
    s.feed("Done — here is the result.");
    assert_eq!(s.finish(), "Done — here is the result.");
}

#[test]
fn hallucinated_tool_response_wrapper_is_suppressed() {
    // `<tool_response>` is a SERVER-side wrapper the chat template puts
    // around role=tool messages. When the model emits one it is
    // fabricating a tool exchange that never happened — the most
    // dangerous leak class, because it reads as real output.
    let markers = qwen();
    let mut s = Stream::new(&markers);
    s.feed("I read the file. <tool_response>fn add() -> i32 { 41 }</tool_response> It returns 41.");
    let out = s.finish();
    assert_eq!(out, "I read the file.  It returns 41.");
}

#[test]
fn a_leak_that_never_closes_is_dropped_at_end_of_stream() {
    // The model started a fragment and hit EOS. Nothing after the opener
    // may be emitted, and the flush must not release the held bytes.
    let markers = qwen();
    let mut s = Stream::new(&markers);
    s.feed("here goes <parameter=path>/etc/shadow");
    assert!(s.suppressing());
    assert_eq!(s.finish(), "here goes ");
}

#[test]
fn primary_arg_is_client_case_insensitive() {
    // opencode sends lowercase tool names (`bash`, `write`); Claude Code
    // sends Anthropic-style capitals. Both must bucket identically or the
    // same session looks like two different tools depending on client.
    use crate::api::sanitizer::primary_arg_for_tool;
    let lower = primary_arg_for_tool("bash", r#"{"command":"cd /tmp && cargo init"}"#);
    let upper = primary_arg_for_tool("Bash", r#"{"command":"cd /tmp && cargo init"}"#);
    assert_eq!(lower, upper);
    assert_eq!(lower.as_deref(), Some("cargo init"));

    let lower = primary_arg_for_tool("write", r#"{"filePath":"/tmp/x.rs"}"#);
    let upper = primary_arg_for_tool("Write", r#"{"file_path":"/tmp/x.rs"}"#);
    assert_eq!(lower, upper);
    assert_eq!(lower.as_deref(), Some("/tmp/x.rs"));
}
