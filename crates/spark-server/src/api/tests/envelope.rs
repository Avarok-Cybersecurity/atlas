// SPDX-License-Identifier: AGPL-3.0-only

//! F73 (2026-04-29): envelope-aware sanitization.
//!
//! MiniMax wraps its tool calls in `<minimax:tool_call>…</minimax:tool_call>`
//! and xgrammar's TagDispatch is non-anchored across BPE merge boundaries,
//! so the envelope reaches Content in three observed spellings. Inside an
//! envelope the inner `<invoke …>` / `<parameter name=…>` tags are
//! legitimate; outside one they are orphan fragments. Before F73 the
//! sanitizer dropped the inner block either way and `parse_tool_calls`
//! was handed nothing to extract — opencode 9-tool sessions lost every
//! call.

use super::harness::Stream;
use crate::tool_parser::{MinimaxXmlParser, ToolCallParser};

const ENVELOPES: [(&str, &str); 3] = [
    ("<minimax:tool_call>", "</minimax:tool_call>"),
    ("<minimax:_call>", "</minimax:_call>"),
    ("<tool_call>", "</tool_call>"),
];

#[test]
fn every_envelope_spelling_lets_the_inner_call_through() {
    let markers = MinimaxXmlParser.leak_markers();
    for (open, close) in ENVELOPES {
        let body = format!(
            "{open}\n<invoke name=\"bash\">\n<parameter name=\"command\">uname -r</parameter>\n</invoke>\n{close}"
        );
        let mut s = Stream::new(&markers);
        s.feed(&body);
        assert!(
            !s.suppressing(),
            "{open}: the envelope path must not enter orphan suppression"
        );
        assert!(!s.inside_envelope(), "{open}: state cleared after close");
        // The envelope bytes are content too: `parse_tool_calls`
        // normalises `<minimax:_call>` to `<tool_call>` downstream and
        // pulls the call out of this same stream.
        assert_eq!(s.finish(), body, "{open}: body must pass through verbatim");
    }
}

#[test]
fn envelope_spellings_survive_byte_at_a_time_chunking() {
    // The BPE-broken spelling exists precisely because the envelope
    // straddles token boundaries, so the fragmented case is the real one.
    let markers = MinimaxXmlParser.leak_markers();
    for (open, close) in ENVELOPES {
        let body = format!("{open}<invoke name=\"read\"></invoke>{close}");
        let mut s = Stream::new(&markers);
        s.feed_chunked(&body, 1);
        assert_eq!(s.finish(), body, "{open}: fragmented envelope");
    }
}

#[test]
fn an_invoke_outside_any_envelope_is_still_suppressed() {
    // Unchanged from the pre-F73 sanitizer: a stray `<invoke …>` with no
    // envelope around it is a hallucinated fragment and is dropped.
    let markers = MinimaxXmlParser.leak_markers();
    let mut s = Stream::new(&markers);
    s.feed("prefix<invoke name=\"bash\">cmd</invoke>tail");
    let out = s.finish();
    assert_eq!(out, "prefixtail");
}

#[test]
fn an_unterminated_envelope_does_not_leave_the_stream_stuck_open() {
    // The model opened an envelope and hit EOS. The already-emitted
    // prefix stands; nothing may be invented to close it.
    let markers = MinimaxXmlParser.leak_markers();
    let mut s = Stream::new(&markers);
    s.feed("thinking<minimax:tool_call><invoke name=\"read\">");
    assert!(
        s.inside_envelope(),
        "an unclosed envelope stays open for the rest of the stream"
    );
    assert!(
        !s.suppressing(),
        "an open envelope must never engage orphan suppression"
    );
    let out = s.finish();
    assert!(
        out.starts_with("thinking<minimax:tool_call>"),
        "prefix and envelope opener emit: {out:?}"
    );
}
