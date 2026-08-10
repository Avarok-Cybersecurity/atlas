// SPDX-License-Identifier: AGPL-3.0-only

//! Complete-tag scrubber for tool-call markup that leaked into content.
//!
//! Split out of `sanitizer.rs` (500-LoC cap): `scrub_tool_tags` is the
//! defense-in-depth pass applied by `stream_guards::flush_content_sanitizer`
//! to end-of-stream tail dumps. It is intentionally NOT part of the
//! per-chunk `sanitize_content_chunk` state machine — that machine already
//! consumes every complete marker on the streaming path, and inside a
//! recognized envelope (F73, minimax) the inner tags are legitimate
//! content the downstream parser extracts.

use crate::tool_parser;

/// Remove any COMPLETE tool-call markup tags from a content string that is
/// about to be emitted to the client. Derived from `markers` so it stays
/// correct for every parser. Handles both exact full tags (`</tool_call>`,
/// `<tool_call>`, `</_call>`) and attribute-prefix opens (`<function=…>`,
/// `<parameter=…>`) by removing through their closing `>`. A partial tag
/// with no closing `>` ends the scan (its remainder is a tag still being
/// formed — the caller's hold-back buffer retains real partials, so the
/// only way one reaches here is a desync dump, where dropping it is correct).
pub(crate) fn scrub_tool_tags(text: &str, markers: &tool_parser::LeakMarkers) -> String {
    if text.is_empty() || (markers.orphan_open.is_empty() && markers.close.is_empty()) {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    'outer: while i < text.len() {
        if bytes[i] == b'<' {
            // Exact full-tag markers (close tags + full-tag opens that
            // already carry their own `>`).
            for m in markers.close.iter().chain(markers.orphan_open.iter()) {
                if m.ends_with('>') && text[i..].starts_with(m) {
                    i += m.len();
                    continue 'outer;
                }
            }
            // Attribute-prefix opens (e.g. `<function=`, `<parameter=`,
            // `<param=`): remove from the prefix through the next `>`.
            for m in markers.orphan_open.iter() {
                if !m.ends_with('>') && text[i..].starts_with(m) {
                    match text[i..].find('>') {
                        Some(gt) => {
                            i += gt + 1;
                            continue 'outer;
                        }
                        // Partial open tag still forming at end-of-text —
                        // drop the remainder (a desync fragment).
                        None => return out,
                    }
                }
            }
        }
        let ch_len = text[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
        out.push_str(&text[i..i + ch_len]);
        i += ch_len;
    }
    out
}
