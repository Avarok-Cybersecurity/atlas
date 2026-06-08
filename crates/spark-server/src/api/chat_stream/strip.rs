// SPDX-License-Identifier: AGPL-3.0-only
//
// Boundary-preserving string strip helper for the thinking-phase
// emit pipeline.
//
// Motivation (2026-05-23 sweep): handle_token's `<think>` strippers
// previously concatenated `before + after` whenever they removed a
// tag (`<tool_call>...</tool_call>`, `<function=…`, `</parameter>`,
// `</function>`, `</tool_call>`, `useruser`-style role-word pairs).
// When the model emitted XML markers mid-word inside `<think>`,
// the strip glued the surrounding words together —
// opencode-session.md captured `directoryis`, `createdirectories`,
// `theserver`, `alreadyrunning` and similar concatenated-word
// artifacts at lines 437, 469, 1262, 1433.
//
// `strip_preserving_boundary` rebuilds the splice with a single
// space inserted iff neither side already provides whitespace.

/// Strip the `[start..end_exclusive)` byte range from `s`, inserting
/// a single space at the splice point when removal would otherwise
/// glue two non-whitespace characters together. See module docs.
pub(super) fn strip_preserving_boundary(s: &str, start: usize, end_exclusive: usize) -> String {
    debug_assert!(start <= end_exclusive && end_exclusive <= s.len());
    let before = &s[..start];
    let after = &s[end_exclusive..];
    let needs_space = !before.is_empty()
        && !after.is_empty()
        && !before.ends_with(char::is_whitespace)
        && !after.starts_with(char::is_whitespace);
    if needs_space {
        format!("{before} {after}")
    } else {
        format!("{before}{after}")
    }
}

/// Boundary-preserving variant of `s.replace(tag, "")` — strips every
/// occurrence of `tag` and inserts a space at the splice when needed
/// to keep word boundaries intact.
pub(super) fn strip_all_preserving_boundary(s: &str, tag: &str) -> String {
    let mut out = s.to_string();
    while let Some(pos) = out.find(tag) {
        out = strip_preserving_boundary(&out, pos, pos + tag.len());
    }
    out
}

/// Env-gated diagnostic trace for the thinking-phase emit path.
/// Off by default; logs at DEBUG when `ATLAS_THINKING_DECODE_TRACE` is
/// set in the environment. Lets us spot strip-loop mutations against
/// the raw decoded delta when investigating residual missing-space
/// artifacts in `<think>` blocks.
pub(super) fn maybe_log_decode_trace(raw: &str, cleaned: &str, full_len: usize, emitted_in: usize) {
    if std::env::var_os("ATLAS_THINKING_DECODE_TRACE").is_none() {
        return;
    }
    let raw_head: String = raw.chars().take(64).collect();
    let cleaned_head: String = cleaned.chars().take(64).collect();
    tracing::debug!(
        full_len,
        emitted_in,
        raw_len = raw.len(),
        cleaned_len = cleaned.len(),
        mutated = raw != cleaned,
        raw = %raw_head,
        cleaned = %cleaned_head,
        "thinking-decode-trace"
    );
}

#[cfg(test)]
mod tests {
    use super::strip_preserving_boundary;

    #[test]
    fn inserts_space_between_glued_words() {
        // "the<tool_call>...</tool_call>project" → "the project"
        let s = "the<tool_call>foo</tool_call>project";
        let start = s.find("<tool_call>").unwrap();
        let end = s.find("</tool_call>").unwrap() + "</tool_call>".len();
        assert_eq!(strip_preserving_boundary(s, start, end), "the project");
    }

    #[test]
    fn no_double_space_when_already_separated() {
        // Both sides already whitespace-adjacent — no extra space.
        let s = "before <tool_call>foo</tool_call> after";
        let start = s.find("<tool_call>").unwrap();
        let end = s.find("</tool_call>").unwrap() + "</tool_call>".len();
        assert_eq!(strip_preserving_boundary(s, start, end), "before  after");
    }

    #[test]
    fn no_space_when_after_starts_with_newline() {
        let s = "context</parameter>\nnext line";
        let start = s.find("</parameter>").unwrap();
        let end = start + "</parameter>".len();
        assert_eq!(
            strip_preserving_boundary(s, start, end),
            "context\nnext line"
        );
    }

    #[test]
    fn empty_before_is_safe() {
        let s = "<think>hello";
        let start = 0;
        let end = "<think>".len();
        assert_eq!(strip_preserving_boundary(s, start, end), "hello");
    }
}
