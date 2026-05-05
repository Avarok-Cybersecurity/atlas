// SPDX-License-Identifier: AGPL-3.0-only

//! F12 tool-call cap + loop watchdog helpers, hoisted from `duplicate.rs`
//! to keep that file under the 500 LoC cap.
//!
//! These two helpers are independent of the F49/F50 duplicate-write pipeline
//! that owns `duplicate.rs`; they live as siblings rather than peers so the
//! file split is invisible to callers (re-exported through `failures/mod.rs`).

/// F12 (2026-04-26): bump the per-response tool-call counter and
/// trip `stop_string_triggered` when the cap is exceeded. Catches
/// pathological responses emitting dozens of tool calls (observed
/// under heavy looping). Default cap = 12 (env override
/// `ATLAS_MAX_TOOL_CALLS_PER_RESPONSE`); well below any legitimate
/// burst (Anthropic's pre-regression default ceiling was 60+).
pub fn bump_f12_tool_call_count(count: &mut usize, max: usize, stop: &mut bool) {
    *count += 1;
    if *count > max && !*stop {
        tracing::warn!(
            emitted = *count,
            max,
            "F12: tool-call cap reached; ending response"
        );
        *stop = true;
    }
}

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
    // Substring fallback: catches a phrase that recurs whole even
    // when one occurrence is glued onto another line (mid-stream
    // narration ramping). Only count for ≥30-char phrases so we
    // don't false-positive on short common fragments.
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
