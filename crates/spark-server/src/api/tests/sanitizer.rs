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

mod sanitizer_tests {
    use super::flush_content_sanitizer;
    use crate::tool_parser::{LeakMarkers, Qwen3CoderParser, ToolCallParser};

    /// F73 (2026-04-29): test wrapper that defaults the new
    /// `inside_envelope: &mut bool` parameter. Tests in this module
    /// that pre-date F73 don't exercise envelope semantics — they
    /// either use `LeakMarkers::EMPTY` (no envelope markers) or the
    /// Qwen3-coder marker set (no envelope_open/close). Either way,
    /// inside_envelope stays false throughout. Keeping the wrapper
    /// avoids per-test mechanical churn.
    fn sanitize_content_chunk(
        text: &str,
        tag_scan_buf: &mut String,
        suppressing_param_leak: &mut bool,
        markers: &LeakMarkers,
    ) -> String {
        let mut inside_envelope = false;
        super::sanitize_content_chunk(
            text,
            tag_scan_buf,
            suppressing_param_leak,
            &mut inside_envelope,
            markers,
        )
    }

    /// F73 (2026-04-29): inner `<invoke ...></invoke>` block passes
    /// through unsuppressed when wrapped in any of the three
    /// recognised MiniMax envelope forms (canonical, BPE-broken,
    /// rewritten). Verifies the live failure mode where opencode
    /// 9-tool sessions emitted `<minimax:_call>...<invoke ...>
    /// </invoke>...</minimax:_call>` and the prior sanitizer
    /// dropped the inner block.
    #[test]
    fn sanitizer_envelope_open_disables_orphan_suppression() {
        // Use MinimaxXmlParser's markers via the trait so the test
        // tracks what the parser actually exports.
        let markers = crate::tool_parser::MinimaxXmlParser.leak_markers();

        for envelope_open in &["<minimax:tool_call>", "<minimax:_call>", "<tool_call>"] {
            let envelope_close = match *envelope_open {
                "<minimax:tool_call>" => "</minimax:tool_call>",
                "<minimax:_call>" => "</minimax:_call>",
                _ => "</tool_call>",
            };
            let body = format!(
                "{envelope_open}\n<invoke name=\"bash\">\n<parameter name=\"command\">uname -r</parameter>\n</invoke>\n{envelope_close}"
            );
            let mut buf = String::new();
            let mut suppress = false;
            let mut env = false;
            let out =
                super::sanitize_content_chunk(&body, &mut buf, &mut suppress, &mut env, &markers);
            // Inner content + envelope tags survive — the parser
            // downstream extracts the tool call from this stream.
            assert!(
                out.contains("<invoke name=\"bash\">"),
                "envelope {envelope_open}: <invoke> must survive: out={out:?}"
            );
            assert!(
                out.contains("uname -r"),
                "envelope {envelope_open}: command must survive: out={out:?}"
            );
            assert!(
                out.contains("</invoke>"),
                "envelope {envelope_open}: </invoke> must survive: out={out:?}"
            );
            // Envelope markers themselves are content too — the
            // parser normalises `<minimax:_call>` → `<tool_call>`
            // downstream and pulls out the inner block.
            assert!(
                out.contains(envelope_open),
                "envelope_open bytes must pass through: out={out:?}"
            );
            assert!(
                out.contains(envelope_close),
                "envelope_close bytes must pass through: out={out:?}"
            );
            assert!(!suppress, "envelope path must not enter orphan suppression");
            // After envelope_close the flag is back to false.
            assert!(!env, "envelope state cleared after close");
        }
    }

    /// F73 (2026-04-29): orphan-suppression behaviour preserved when
    /// `<invoke ...>` appears OUTSIDE any envelope. Unchanged from the
    /// pre-F73 sanitizer for a stray-fragment hallucination case.
    #[test]
    fn sanitizer_orphan_invoke_outside_envelope_still_suppressed() {
        let markers = crate::tool_parser::MinimaxXmlParser.leak_markers();
        let body = "prefix<invoke name=\"bash\">cmd</invoke>tail";
        let mut buf = String::new();
        let mut suppress = false;
        let mut env = false;
        let out = super::sanitize_content_chunk(body, &mut buf, &mut suppress, &mut env, &markers);
        assert!(
            out.starts_with("prefix"),
            "non-orphan prefix emits: {out:?}"
        );
        assert!(
            !out.contains("<invoke"),
            "stray <invoke> must still be suppressed: {out:?}"
        );
        assert!(
            !out.contains("cmd"),
            "suppressed body bytes must not leak: {out:?}"
        );
    }

    #[test]
    fn sanitizer_noop_for_empty_markers() {
        // A parser that opts out (Hermes, Gemma4, Mistral, BareJson)
        // passes text through verbatim. No buffering, no latency tail.
        let mut buf = String::new();
        let mut suppress = false;
        let out = sanitize_content_chunk(
            "<parameter=foo>value</parameter>",
            &mut buf,
            &mut suppress,
            &LeakMarkers::EMPTY,
        );
        assert_eq!(out, "<parameter=foo>value</parameter>");
        assert!(buf.is_empty(), "no markers → no tail buffering");
        assert!(!suppress);
    }

    #[test]
    fn sanitizer_suppresses_for_qwen3_markers() {
        // Existing Qwen3-coder behaviour via trait-delivered markers.
        // The orphan `<parameter=...>VALUE</parameter>` block is dropped
        // entirely; only the bytes outside the leak survive.
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::new();
        let mut suppress = false;
        let out = sanitize_content_chunk(
            "prefix<parameter=filePath>/tmp/x.txt</parameter>suffix</function>tail",
            &mut buf,
            &mut suppress,
            &markers,
        );
        // "prefix" emits; the `<parameter=filePath>...</parameter>` body
        // is suppressed; the stray `</function>` is dropped; "tail" is
        // short enough to stay buffered (no trailing tag-chars).
        assert!(out.starts_with("prefix"), "got: {out:?}");
        assert!(
            !out.contains("<parameter="),
            "orphan open must not leak: {out:?}"
        );
        assert!(
            !out.contains("/tmp/x.txt"),
            "suppressed body must not leak: {out:?}"
        );
        assert!(
            !out.contains("</function>"),
            "stray close must be stripped: {out:?}"
        );
    }

    #[test]
    fn sanitizer_fuses_tag_across_chunks() {
        // The whole point of the tail buffer: a tag arriving split
        // across two calls still matches. The first chunk is shorter
        // than (tag_max - 1), so nothing is emitted yet — we cannot
        // prove the `<param` suffix is not a tag prefix.
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::new();
        let mut suppress = false;
        let out1 = sanitize_content_chunk("abc<param", &mut buf, &mut suppress, &markers);
        assert!(!suppress, "partial tag must not trigger suppression");
        assert_eq!(out1, "", "short chunk stays in tail buffer awaiting fusion");
        let out2 = sanitize_content_chunk(
            "eter=x>body</parameter>tail",
            &mut buf,
            &mut suppress,
            &markers,
        );
        // Fusion: `<parameter=x>` found in the combined buffer.
        // "abc" prefix emits; body suppressed; `</parameter>` ends
        // suppression; "tail" stays buffered (too short to flush).
        assert!(
            out2.starts_with("abc"),
            "prefix emits after fusion: {out2:?}"
        );
        assert!(
            !out2.contains("body"),
            "suppressed body must not leak: {out2:?}"
        );
        assert!(
            !out2.contains("<parameter="),
            "orphan open must not leak: {out2:?}"
        );
        assert!(!suppress, "close tag exits suppression state");
    }

    #[test]
    fn flush_empty_markers_emits_tail_verbatim() {
        // With EMPTY markers the fast path never buffers, but the flush
        // must still handle any residual correctly (it should always be
        // empty in practice).
        let mut buf = String::from("anything");
        let mut suppress = false;
        let out = flush_content_sanitizer(&mut buf, &mut suppress, &LeakMarkers::EMPTY);
        assert_eq!(out, "anything");
        assert!(buf.is_empty());
    }

    #[test]
    fn flush_drops_partial_tag_prefix() {
        // A bare `<par` tail could fuse into `<parameter=` on a next
        // chunk, but stream ended — drop it to avoid emitting mid-tag.
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::from("<par");
        let mut suppress = false;
        let out = flush_content_sanitizer(&mut buf, &mut suppress, &markers);
        assert_eq!(out, "");
    }

    /// F73 gate on the flush-time scrub: envelope-capable parsers
    /// (minimax) legitimately stream envelope + inner tool tags as
    /// content — the downstream parser extracts the call from them.
    /// The flush must NOT scrub complete markers for such parsers.
    #[test]
    fn flush_envelope_markers_skips_scrub() {
        let markers = crate::tool_parser::MinimaxXmlParser.leak_markers();
        let tail = "</invoke>\n</minimax:tool_call>";
        let mut buf = String::from(tail);
        let mut suppress = false;
        let out = flush_content_sanitizer(&mut buf, &mut suppress, &markers);
        assert_eq!(out, tail, "envelope content must survive flush verbatim");
    }

    // Note: the bash-fence tool-call salvage stack was removed (the
    // model now emits clean tool calls via the grammar fix), so its
    // tests no longer exist.
    //
    // Note: the `strip_xml_leaks_from_assistant_content` tests were
    // removed when that helper was deleted in #90 (the model now emits
    // clean tool calls via the grammar fix).

    // Note: the bare-XML tool-call salvage stack was removed (the model
    // now emits clean tool calls via the grammar fix), so its tests no
    // longer exist.

    #[test]
    fn flush_before_tool_boundary_recovers_from_stuck_suppression() {
        // Simulates the production bug: model emits `<parameter=` in
        // prose (sanitizer enters suppression), then a real structured
        // tool call arrives and its `</parameter>` is consumed by the
        // detector — never reaching the sanitizer. Without the pre-tool
        // flush introduced alongside this test, `suppressing_param_leak`
        // would stay `true` forever and eat all post-tool content.
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::new();
        let mut suppress = false;

        // Step 1: prose orphan triggers suppression.
        let prose = sanitize_content_chunk(
            "Let me write it: <parameter=content>foo",
            &mut buf,
            &mut suppress,
            &markers,
        );
        assert_eq!(prose, "Let me write it: ", "prefix emits: {prose:?}");
        assert!(suppress, "orphan `<parameter=` enters suppression");

        // Step 2: simulate Content → Tool boundary (detector emits Tool
        // event). Our fix calls flush here.
        let pre_tool = flush_content_sanitizer(&mut buf, &mut suppress, &markers);
        assert_eq!(pre_tool, "", "suppressed tail is correctly dropped");
        assert!(!suppress, "flush clears the suppression flag");
        assert!(buf.is_empty(), "flush clears the tail buffer");

        // Step 3: post-tool content must flow through — this is the
        // regression we're pinning.
        let post_tool = sanitize_content_chunk(
            "Done — here is the result.",
            &mut buf,
            &mut suppress,
            &markers,
        );
        assert!(
            post_tool.starts_with("Done"),
            "post-tool content must reach the client: {post_tool:?}"
        );
        assert!(!suppress, "no new orphan, must stay out of suppression");
    }

    // Note: the prose→Write tool-call salvage stack was removed (the
    // model now emits clean tool calls via the grammar fix), so its
    // tests no longer exist.
    //
    // Note: cross-turn prose-prefix Layer 4 was deleted along with
    // its `normalise_text_prefix` helper; the unified loop detector
    // in `crate::loop_detector` covers the same ground via shingle
    // similarity over assistant text. See `loop_detector.rs` tests
    // (`three_identical_intros_fire_loop`,
    // `slightly_varied_intros_still_fire`).
}

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
    assert_eq!(
        first, "",
        "a chunk that could still be a tag prefix must not be emitted yet"
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
