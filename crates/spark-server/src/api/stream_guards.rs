// SPDX-License-Identifier: AGPL-3.0-only

//! Streaming-side runtime guards. These are NOT prompt-mutating
//! injections — they operate purely on outbound model text:
//!
//! - [`bump_f12_tool_call_count`]: caps the number of tool calls
//!   emitted per response (default 12) so a degenerate streamer
//!   can't dump dozens.
//! - [`check_loop_watchdog`]: detects a repeating line/phrase in
//!   the post-detector content stream and signals end-of-response.
//! - [`flush_content_sanitizer`]: drains the tag-scan tail buffer
//!   when the stream closes, suppressing incomplete tag openers.
//!
//! Restored from the deleted `api/failures/` subtree (Phase A,
//! 2026-05-24): the subtree's job was prompt-injection helpers
//! (F7/F23/F29/...) which were removed wholesale, but these three
//! streaming guards were sibling-hosted there only because of an
//! old file-split. They have no input/prompt side and remain.

use crate::tool_parser;

/// Bump the per-response tool-call counter and trip
/// `stop_string_triggered` when the cap is exceeded. Catches
/// pathological responses emitting dozens of tool calls. Default
/// cap = 12 (env override `ATLAS_MAX_TOOL_CALLS_PER_RESPONSE`).
pub fn bump_f12_tool_call_count(count: &mut usize, max: usize, stop: &mut bool) {
    *count += 1;
    if *count > max && !*stop {
        tracing::warn!(
            emitted = *count,
            max,
            "tool-call cap reached; ending response"
        );
        *stop = true;
    }
}

/// Detect a repeating line or long phrase in the post-detector
/// content buffer. Returns true when the last non-trivial line
/// occurs (fuzzy-matched on collapsed whitespace) at least 4 times
/// in the running 10 KB window, or when a ≥30-char phrase recurs
/// 4 times via substring scan. Caller is expected to stop the
/// stream once true is returned.
pub fn check_loop_watchdog(
    text: &str,
    loop_scan_buf: &mut String,
    already_triggered: bool,
) -> bool {
    if already_triggered || text.is_empty() {
        return false;
    }
    loop_scan_buf.push_str(text);
    if loop_scan_buf.len() > 10_240 {
        let drop = loop_scan_buf.len() - 8_192;
        let cut = loop_scan_buf
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= drop)
            .unwrap_or(drop);
        loop_scan_buf.drain(..cut);
    }
    let last_line = loop_scan_buf
        .lines()
        .rev()
        .find(|l| l.trim().len() > 15 && !l.trim_start().starts_with("```"))
        .map(|s| s.to_string());
    let Some(line) = last_line else {
        return false;
    };
    fn norm(s: &str) -> String {
        let lowered = s.trim().to_ascii_lowercase();
        let mut out = String::with_capacity(lowered.len());
        let mut prev_space = false;
        for ch in lowered.chars() {
            if ch.is_ascii_whitespace() {
                if !prev_space && !out.is_empty() {
                    out.push(' ');
                }
                prev_space = true;
            } else {
                out.push(ch);
                prev_space = false;
            }
        }
        if out.ends_with(' ') {
            out.pop();
        }
        out
    }
    let needle = norm(&line);
    if needle.is_empty() {
        return false;
    }
    let exact_occurrences = loop_scan_buf.lines().filter(|l| norm(l) == needle).count();
    if exact_occurrences >= 4 {
        tracing::warn!(
            occurrences = exact_occurrences,
            line_len = needle.len(),
            "loop watchdog fired — repeated line (fuzzy-match) in post-detector content"
        );
        return true;
    }
    if needle.len() >= 30 {
        let lowered_buf = loop_scan_buf.to_ascii_lowercase();
        let mut count = 0usize;
        let mut start = 0usize;
        while let Some(rel) = lowered_buf[start..].find(&needle) {
            count += 1;
            start += rel + needle.len();
            if count >= 4 {
                break;
            }
        }
        if count >= 4 {
            tracing::warn!(
                occurrences = count,
                line_len = needle.len(),
                "loop watchdog fired — repeated phrase (substring) in post-detector content"
            );
            return true;
        }
    }
    false
}

/// Flush anything held in the streaming sanitizer's tail buffer at
/// stream end. Drops content if tag-suppression is still active (no
/// close arrived) or if the remaining bytes look like an incomplete
/// tag opener.
pub fn flush_content_sanitizer(
    tag_scan_buf: &mut String,
    suppressing_param_leak: &mut bool,
    markers: &tool_parser::LeakMarkers,
) -> String {
    if *suppressing_param_leak {
        tag_scan_buf.clear();
        *suppressing_param_leak = false;
        return String::new();
    }
    if tag_scan_buf.is_empty() {
        return String::new();
    }
    let tag_max: usize = markers
        .orphan_open
        .iter()
        .chain(markers.close.iter())
        .map(|t| t.len())
        .max()
        .unwrap_or(0);
    let final_text = std::mem::take(tag_scan_buf);
    let looks_like_partial_tag = {
        let t = final_text.trim_end();
        tag_max > 0 && t.starts_with('<') && !t.contains(char::is_whitespace) && t.len() < tag_max
    };
    if looks_like_partial_tag {
        String::new()
    } else {
        final_text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f12_under_cap_does_not_stop() {
        let (mut count, mut stop) = (0usize, false);
        bump_f12_tool_call_count(&mut count, 12, &mut stop);
        assert_eq!(count, 1);
        assert!(!stop);
    }

    #[test]
    fn f12_at_cap_does_not_stop() {
        let (mut count, mut stop) = (11usize, false);
        bump_f12_tool_call_count(&mut count, 12, &mut stop);
        assert_eq!(count, 12);
        assert!(!stop);
    }

    #[test]
    fn f12_over_cap_trips_stop() {
        let (mut count, mut stop) = (12usize, false);
        bump_f12_tool_call_count(&mut count, 12, &mut stop);
        assert_eq!(count, 13);
        assert!(stop);
    }

    #[test]
    fn watchdog_already_triggered_returns_false() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog("anything", &mut buf, true));
    }

    #[test]
    fn watchdog_empty_text_returns_false() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog("", &mut buf, false));
    }

    #[test]
    fn watchdog_four_identical_lines_fires() {
        let mut buf = String::new();
        let line = "Running cargo test on the project\n";
        assert!(!check_loop_watchdog(line, &mut buf, false));
        assert!(!check_loop_watchdog(line, &mut buf, false));
        assert!(!check_loop_watchdog(line, &mut buf, false));
        assert!(check_loop_watchdog(line, &mut buf, false));
    }

    #[test]
    fn watchdog_buffer_caps_at_10kb() {
        let mut buf = String::new();
        let big = "x".repeat(5000);
        check_loop_watchdog(&big, &mut buf, false);
        check_loop_watchdog(&big, &mut buf, false);
        check_loop_watchdog(&big, &mut buf, false);
        assert!(buf.len() <= 10_240, "buffer should self-trim, got {}", buf.len());
    }
}
