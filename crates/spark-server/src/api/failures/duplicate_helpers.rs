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

pub const HTML_RESTART_GUARD_MIN_BYTES: usize = 500;

pub fn check_loop_watchdog(
    text: &str,
    loop_scan_buf: &mut String,
    already_triggered: bool,
) -> bool {
    check_loop_watchdog_with_context(text, loop_scan_buf, already_triggered, false)
}

pub fn check_loop_watchdog_with_context(
    text: &str,
    loop_scan_buf: &mut String,
    already_triggered: bool,
    code_context_hint: bool,
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
    let lowered_scan_buf = loop_scan_buf.to_ascii_lowercase();
    let html_doc_starts = lowered_scan_buf.matches("<!doctype html").count();
    if html_doc_starts >= 2 {
        let previous_doc_len = completed_html_doc_len_before_last_start(&lowered_scan_buf);
        let previous_doc_is_substantial =
            previous_doc_len.is_some_and(|len| len >= HTML_RESTART_GUARD_MIN_BYTES);
        let repeated_restarts = html_doc_starts >= 3;
        if previous_doc_is_substantial || repeated_restarts {
            tracing::warn!(
                occurrences = html_doc_starts,
                previous_doc_len = previous_doc_len.unwrap_or(0),
                "loop watchdog fired — repeated HTML document start in post-detector content"
            );
            return true;
        }
    }
    let structural_restarts = count_structural_close_open_cycles(&lowered_scan_buf);
    // Only fire the structural restart check when there's a substantial
    // complete document in the buffer. Short drafts should be allowed to
    // restart without triggering the watchdog.
    if structural_restarts >= 1
        && last_complete_html_doc(&lowered_scan_buf)
            .is_some_and(|(_, doc_len)| doc_len >= HTML_RESTART_GUARD_MIN_BYTES)
    {
        tracing::warn!(
            occurrences = structural_restarts,
            "loop watchdog fired — structural restart cycle (close→prose→open) after complete document"
        );
        return true;
    }
    if let Some((html_end, html_doc_len)) = last_complete_html_doc(&lowered_scan_buf)
        && html_doc_len >= HTML_RESTART_GUARD_MIN_BYTES
    {
        let after_html = &lowered_scan_buf[html_end..];
        let after_fence =
            after_html.trim_start_matches(|c: char| c.is_ascii_whitespace() || c == '`');
        // Language-agnostic prose detection: any Unicode alphabetic
        // character after a closing fence signals natural-language
        // self-critique (works for English, Chinese, Arabic, etc.).
        // Structural continuations (HTML tags, code fences, JSON)
        // are distinguished by their opening characters.
        let is_structural = after_fence.starts_with('<')
            || after_fence.starts_with("```")
            || after_fence.starts_with('{')
            || after_fence.starts_with('[');
        let prose_after_complete_html = after_fence
            .chars()
            .next()
            .is_some_and(|ch| ch.is_alphabetic())
            && !is_structural;
        let html_restart = after_html.contains("```html");
        if prose_after_complete_html || html_restart {
            tracing::warn!("loop watchdog fired — HTML response restarted after complete document");
            return true;
        }
    }
    if let Some(occurrences) = css_property_chain_loop_count(&lowered_scan_buf) {
        tracing::warn!(
            occurrences,
            "loop watchdog fired — malformed CSS property-chain loop"
        );
        return true;
    }
    let malformed_d3_attr_count = lowered_scan_buf.matches(".attr(\"transform\") =>").count()
        + lowered_scan_buf.matches(".attr(\"transform\")=>").count();
    if malformed_d3_attr_count >= 3 {
        tracing::warn!(
            occurrences = malformed_d3_attr_count,
            "loop watchdog fired — malformed D3 attr transform loop"
        );
        return true;
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
    // Code blocks contain legitimate short-line repetition (CSS one-liners
    // like `position: absolute;`, `top: 0;` recurring across selectors;
    // JS `let x = ...;` chains). Detect by counting ``` markers in the
    // scan buffer — odd count means the most recent fence opened a code
    // block that hasn't closed yet, so raise the threshold to avoid
    // false-positiving on legitimate code structure. The 4-occurrence
    // threshold still applies outside code blocks (prose loops).
    //
    // 16 inside code blocks chosen empirically (2026-05-11): observed
    // legitimate CSS using `position: absolute;` across 8 selectors in a
    // flight-sim app. 16 gives 2× headroom while still catching genuine
    // attractor loops (which typically repeat dozens of times). Real
    // attractor cases (CSS `.missile { fill: #FF0000 }` × 226) trigger
    // long before 16 if Leviathan's loop-breaking somehow fails to
    // engage.
    let fence_count = loop_scan_buf.matches("```").count();
    let inside_code_block = fence_count % 2 == 1;
    // Some models emit HTML/CSS/JS RAW without wrapping in ``` fences,
    // so the inside_code_block flag misses them and the watchdog
    // false-positives on CSS one-liners like `position: absolute;`.
    // Detect code-like content directly on the candidate line: any
    // bracket/brace/semicolon or a JS/CSS keyword start counts. Mirrors
    // the SimHash code-block detection at the streaming layer.
    let looks_like_code = {
        let l = needle.as_str();
        l.contains('<')
            || l.contains('>')
            || l.contains('{')
            || l.contains('}')
            || l.contains(';')
            || l.contains("=>")
            || l.contains(": ")
            || l.starts_with("const ")
            || l.starts_with("let ")
            || l.starts_with("var ")
            || l.starts_with("function ")
            || l.starts_with("class ")
            || l.starts_with("return ")
            || l.starts_with("import ")
            || l.starts_with("def ")
    };
    let html_context = lowered_scan_buf.contains("<!doctype html")
        || lowered_scan_buf.contains("<html")
        || lowered_scan_buf.contains("<div");
    let code_context = code_context_hint || inside_code_block || looks_like_code || html_context;
    let looks_like_markdown_prose = {
        let l = needle.as_str();
        let numbered_item = l.split_once(". ").is_some_and(|(prefix, _)| {
            !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit())
        });
        l.starts_with('#')
            || l.starts_with("* ")
            || l.starts_with("- ")
            || l.starts_with("+ ")
            || numbered_item
    };
    // Inside an explicit markdown ``` fence we tighten the threshold to
    // 8 occurrences — real code recurs 4-8× legitimately; the dense
    // 27B-FP8 long-context loop emits 16-32× near-identical lines
    // inside fences. Raw HTML/CSS without fences stays at 16 because
    // structural markup naturally recurs there (e.g. 8 CSS selectors).
    let exact_threshold = if inside_code_block && !looks_like_markdown_prose {
        8
    } else if code_context && !looks_like_markdown_prose {
        16
    } else {
        4
    };
    let exact_occurrences = loop_scan_buf.lines().filter(|l| norm(l) == needle).count();
    if exact_occurrences >= exact_threshold {
        tracing::warn!(
            occurrences = exact_occurrences,
            line_len = needle.len(),
            inside_code_block,
            looks_like_code,
            html_context,
            looks_like_markdown_prose,
            "loop watchdog fired — repeated line (fuzzy-match) in post-detector content"
        );
        return true;
    }
    // Substring fallback: catches a phrase that recurs whole even
    // when one occurrence is glued onto another line (mid-stream
    // narration ramping). Only count for ≥30-char phrases so we
    // don't false-positive on short common fragments. Inside code
    // blocks AND inside HTML-generation context, 30+ char "phrases"
    // are legitimate structural patterns that recur across similar
    // sections — apply the same 16× tolerance to avoid false-
    // positiving on real HTML/CSS/JS content that the model emits
    // without ``` fences (where inside_code_block would be false
    // but the content is clearly structured markup).
    if needle.len() >= 30 {
        let substring_threshold = if inside_code_block && !looks_like_markdown_prose {
            8
        } else if code_context && !looks_like_markdown_prose {
            16
        } else {
            4
        };
        let mut count = 0usize;
        let mut start = 0usize;
        while let Some(rel) = lowered_scan_buf[start..].find(&needle) {
            count += 1;
            start += rel + needle.len();
            if count >= substring_threshold {
                break;
            }
        }
        if count >= substring_threshold {
            tracing::warn!(
                occurrences = count,
                line_len = needle.len(),
                inside_code_block,
                looks_like_code,
                html_context,
                looks_like_markdown_prose,
                "loop watchdog fired — repeated phrase (substring) in post-detector content"
            );
            return true;
        }
    }
    false
}

fn last_complete_html_doc(scan_buf: &str) -> Option<(usize, usize)> {
    let html_end_start = scan_buf.rfind("</html>")?;
    let html_end = html_end_start + "</html>".len();
    let before_end = &scan_buf[..html_end_start];
    let html_start = before_end
        .rfind("<!doctype html")
        .or_else(|| before_end.rfind("<html"))?;
    Some((html_end, html_end - html_start))
}

fn completed_html_doc_len_before_last_start(scan_buf: &str) -> Option<usize> {
    let first_start = scan_buf.find("<!doctype html")?;
    let last_start = scan_buf.rfind("<!doctype html")?;
    if first_start == last_start {
        return None;
    }
    let before_last_start = &scan_buf[first_start..last_start];
    let html_end_rel = before_last_start.find("</html>")?;
    Some(html_end_rel + "</html>".len())
}

fn count_structural_close_open_cycles(lowered: &str) -> usize {
    let html_closes: &[&str] = &["</html>"];
    let html_opens: &[&str] = &["<!doctype html", "<html"];
    let mut cycles = 0;
    let mut search_from = 0;
    while let Some(rel) = lowered[search_from..].find(html_closes[0]) {
        let close_end = search_from + rel + html_closes[0].len();
        search_from = close_end;
        let mut found_cycle = false;
        for open in html_opens {
            if let Some(open_rel) = lowered[close_end..].find(open) {
                let between_raw = &lowered[close_end..close_end + open_rel];
                if has_prose_between_html_close_and_open(between_raw) {
                    cycles += 1;
                    found_cycle = true;
                    break;
                }
            }
        }
        if !found_cycle {
            break;
        }
    }
    cycles
}

fn has_prose_between_html_close_and_open(between: &str) -> bool {
    for line in between.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("```") || trimmed.starts_with('<') {
            continue;
        }
        let alpha_run: String = trimmed.chars().filter(|c| c.is_alphabetic()).collect();
        if alpha_run.len() >= 4 {
            return true;
        }
    }
    false
}

fn css_property_chain_loop_count(scan_buf: &str) -> Option<usize> {
    const CSS_PROPS: &[&str] = &[
        "background",
        "border",
        "bottom",
        "color",
        "display",
        "font",
        "height",
        "left",
        "margin",
        "opacity",
        "padding",
        "position",
        "right",
        "top",
        "transform",
        "width",
        "z-index",
    ];
    const THRESHOLD: usize = 8;

    let start = scan_buf
        .char_indices()
        .map(|(i, _)| i)
        .find(|&i| i >= scan_buf.len().saturating_sub(2_048))
        .unwrap_or(0);
    let recent = &scan_buf[start..];
    let tokens = css_loop_tokens(recent);
    let mut occurrences = 0usize;
    for window in tokens.windows(4) {
        if is_css_prop(window[0], CSS_PROPS)
            && window[1] == ":"
            && is_css_prop(window[2], CSS_PROPS)
            && window[3] == ":"
        {
            occurrences += 1;
            if occurrences >= THRESHOLD {
                return Some(occurrences);
            }
        }
    }
    None
}

fn css_loop_tokens(s: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut token_start: Option<usize> = None;
    for (idx, ch) in s.char_indices() {
        if ch.is_ascii_alphanumeric() || ch == '-' {
            token_start.get_or_insert(idx);
            continue;
        }
        if let Some(start) = token_start.take() {
            tokens.push(&s[start..idx]);
        }
        if matches!(ch, ':' | ';' | '{' | '}') {
            tokens.push(&s[idx..idx + ch.len_utf8()]);
        }
    }
    if let Some(start) = token_start {
        tokens.push(&s[start..]);
    }
    tokens
}

fn is_css_prop(token: &str, props: &[&str]) -> bool {
    props.binary_search(&token).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── bump_f12_tool_call_count ──────────────────────────────────

    #[test]
    fn f12_under_cap_does_not_stop() {
        let (mut count, mut stop) = (0usize, false);
        bump_f12_tool_call_count(&mut count, 12, &mut stop);
        assert_eq!(count, 1);
        assert!(!stop);
    }

    #[test]
    fn f12_at_cap_does_not_stop() {
        // The check is `> max`, so count == max is allowed.
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
    fn f12_already_stopped_still_counts() {
        // Even when stop is already set, count keeps incrementing
        // for the diagnostic — but doesn't re-warn.
        let (mut count, mut stop) = (100usize, true);
        bump_f12_tool_call_count(&mut count, 12, &mut stop);
        assert_eq!(count, 101);
        assert!(stop);
    }

    // ── check_loop_watchdog ───────────────────────────────────────

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
    fn watchdog_single_line_returns_false() {
        let mut buf = String::new();
        // Just one line of content — no repeats.
        assert!(!check_loop_watchdog(
            "this is a single long enough line to qualify\n",
            &mut buf,
            false
        ));
    }

    #[test]
    fn watchdog_four_identical_lines_fires() {
        let mut buf = String::new();
        let line = "Running cargo test on the project\n";
        // First three accumulations should not fire.
        assert!(!check_loop_watchdog(line, &mut buf, false));
        assert!(!check_loop_watchdog(line, &mut buf, false));
        assert!(!check_loop_watchdog(line, &mut buf, false));
        // Fourth occurrence trips the watchdog.
        assert!(check_loop_watchdog(line, &mut buf, false));
    }

    #[test]
    fn watchdog_fuzzy_normalization_collapses_whitespace() {
        let mut buf = String::new();
        // Same phrase, different whitespace each time — must still fuzzy-match.
        assert!(!check_loop_watchdog(
            "Running cargo test now\n",
            &mut buf,
            false
        ));
        assert!(!check_loop_watchdog(
            "  Running cargo test now  \n",
            &mut buf,
            false
        ));
        assert!(!check_loop_watchdog(
            "Running cargo  test  now\n",
            &mut buf,
            false
        ));
        assert!(check_loop_watchdog(
            "Running\tcargo test now\n",
            &mut buf,
            false
        ));
    }

    #[test]
    fn watchdog_short_lines_skipped() {
        // Lines whose trimmed length ≤ 15 chars don't qualify as the
        // needle, so identical short lines don't trigger.
        let mut buf = String::new();
        let short = "ok\n"; // 2 chars
        for _ in 0..10 {
            assert!(!check_loop_watchdog(short, &mut buf, false));
        }
    }

    #[test]
    fn watchdog_buffer_caps_at_10kb() {
        let mut buf = String::new();
        let big = "x".repeat(5000);
        check_loop_watchdog(&big, &mut buf, false);
        check_loop_watchdog(&big, &mut buf, false);
        // After two 5KB pushes the buffer is 10KB; a third triggers the
        // 10_240-byte cap and drains down to 8KB.
        check_loop_watchdog(&big, &mut buf, false);
        assert!(
            buf.len() <= 10_240,
            "buffer should self-trim, got {}",
            buf.len()
        );
    }

    #[test]
    fn watchdog_allows_repeated_raw_css_lines_until_code_threshold() {
        let mut buf = String::new();
        let line = "position: absolute;\n";
        for _ in 0..15 {
            assert!(!check_loop_watchdog(line, &mut buf, false));
        }
        assert!(check_loop_watchdog(line, &mut buf, false));
    }

    #[test]
    fn watchdog_allows_repeated_raw_css_phrases_until_code_threshold() {
        let mut buf = String::new();
        let phrase = ".missile { fill: #FF0000; stroke: #AA0000; }\n";
        for _ in 0..15 {
            assert!(!check_loop_watchdog(phrase, &mut buf, false));
        }
        assert!(check_loop_watchdog(phrase, &mut buf, false));
    }

    fn substantial_html_document() -> String {
        format!(
            "```html\n<!DOCTYPE html><html><body><script>{}</script></body></html>\n",
            "x".repeat(HTML_RESTART_GUARD_MIN_BYTES)
        )
    }

    #[test]
    fn watchdog_catches_restarted_html_document() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog(
            &substantial_html_document(),
            &mut buf,
            false
        ));
        assert!(check_loop_watchdog(
            "Let me write this out properly.\n\n```html\n<!DOCTYPE html>\n",
            &mut buf,
            false
        ));
    }

    #[test]
    fn watchdog_allows_restarted_html_after_short_draft() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog(
            "```html\n<!DOCTYPE html><html><body><script></script></body></html>\n",
            &mut buf,
            false
        ));
        assert!(!check_loop_watchdog(
            "\n```\n\nWait, that was only a draft.\n\n```html\n<!DOCTYPE html>\n",
            &mut buf,
            false
        ));
    }

    #[test]
    fn watchdog_catches_structural_restart_cycle() {
        let mut buf = String::new();
        // First complete HTML document (substantial, > 500 bytes)
        let doc = format!(
            "```html\n<!DOCTYPE html><html><body>{}</body></html>\n",
            "x".repeat(HTML_RESTART_GUARD_MIN_BYTES)
        );
        assert!(!check_loop_watchdog(&doc, &mut buf, false));
        // Self-criticism prose + HTML restart: structural restart cycle
        // (</html> → prose → <!DOCTYPE html>)
        assert!(check_loop_watchdog(
            "\n```\n\nWait let me rewrite\n\n```html\n<!DOCTYPE html>\n",
            &mut buf,
            false
        ));
    }

    #[test]
    fn watchdog_catches_structural_restart_cycle_chinese() {
        let mut buf = String::new();
        let doc = format!(
            "```html\n<!DOCTYPE html><html><body>{}</body></html>\n",
            "x".repeat(HTML_RESTART_GUARD_MIN_BYTES)
        );
        assert!(!check_loop_watchdog(&doc, &mut buf, false));
        // Chinese self-criticism prose: 让我重写 (let me rewrite)
        assert!(check_loop_watchdog(
            "\n```\n\n让我重写\n\n```html\n<!DOCTYPE html>\n",
            &mut buf,
            false
        ));
    }

    #[test]
    fn watchdog_catches_malformed_css_property_chain_loop() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog(
            "style=\"left:top:0;width:top:-15;left:top:0;width:top:-15;",
            &mut buf,
            false
        ));
        assert!(check_loop_watchdog(
            "left:top:-20;width:top:0;left:top:15;width:top:-130;",
            &mut buf,
            false
        ));
    }

    #[test]
    fn watchdog_allows_legitimate_repeated_css_properties() {
        let mut buf = String::new();
        let css = "\
.a { left: 0; top: 0; width: 10px; }\n\
.b { left: 4px; top: 8px; width: 12px; }\n\
.c { left: 6px; top: 9px; width: 14px; }\n\
.d { left: 7px; top: 10px; width: 16px; }\n";
        assert!(!check_loop_watchdog(css, &mut buf, false));
    }

    #[test]
    fn watchdog_catches_malformed_d3_attr_transform_loop() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog(
            ".attr(\"transform\") => {\nreturn `translate(${x}, 0)`;\n",
            &mut buf,
            false
        ));
        assert!(!check_loop_watchdog(
            ".attr(\"transform\") => {\nreturn `translate(${x + 20}, 0)`;\n",
            &mut buf,
            false
        ));
        assert!(check_loop_watchdog(
            ".attr(\"transform\") => {\nreturn `translate(${x + 30}, 0)`;\n",
            &mut buf,
            false
        ));
    }

    #[test]
    fn watchdog_catches_self_rewrite_after_complete_html() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog(
            &substantial_html_document(),
            &mut buf,
            false
        ));
        assert!(!check_loop_watchdog("\n```\n\n", &mut buf, false));
        assert!(check_loop_watchdog("T", &mut buf, false));
    }

    #[test]
    fn watchdog_allows_self_rewrite_after_short_complete_html() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog(
            "```html\n<!DOCTYPE html><html><body><script></script></body></html>\n",
            &mut buf,
            false
        ));
        assert!(!check_loop_watchdog("\n```\n\n", &mut buf, false));
        assert!(!check_loop_watchdog(
            "This was only a draft; I should write the complete app.\n",
            &mut buf,
            false
        ));
    }

    #[test]
    fn watchdog_html_context_uses_code_threshold_for_exact() {
        // Repeated phrase inside an HTML document (no ``` fences)
        // should use threshold 16 instead of 4.
        let mut buf = String::new();
        let line = "Here is a detailed explanation of the app\n";
        buf.push_str("<!DOCTYPE html>\n<html>\n<body>\n");
        for _ in 0..14 {
            assert!(!check_loop_watchdog(line, &mut buf, false));
        }
        // 15 occurrences: still under the code-aware threshold (16).
        assert!(!check_loop_watchdog(line, &mut buf, false));
        // 16+ occurrences: now triggers.
        assert!(check_loop_watchdog(line, &mut buf, false));
    }

    #[test]
    fn watchdog_markdown_heading_uses_prose_threshold_in_html_context() {
        // Dense code generations sometimes close a code fence early,
        // then loop on Markdown headings like "### Key Features:".
        // HTML context should not lift those prose headings to the
        // 16-occurrence code threshold.
        let mut buf = String::new();
        buf.push_str("<!DOCTYPE html>\n<html>\n<body>\n");
        let line = "### Key Features:\n";
        for _ in 0..3 {
            assert!(!check_loop_watchdog(line, &mut buf, false));
        }
        assert!(check_loop_watchdog(line, &mut buf, false));
    }

    #[test]
    fn watchdog_html_context_uses_code_threshold_for_substring() {
        // 34-char repeated phrase inside HTML (no ``` fences, not
        // appearance-code-like) should use substring threshold 16
        // instead of the prose threshold 4.
        let mut buf = String::new();
        let phrase = "position:relative;display:flex;align-items:center;\n";
        assert!(phrase.len() >= 30);
        buf.push_str("<!DOCTYPE html>\n<html>\n<body>\n");
        // Build up 3 occurrences — below prose threshold of 4, easily
        // below code threshold of 16.
        for _ in 0..3 {
            assert!(!check_loop_watchdog(phrase, &mut buf, false));
        }
        // 4 occurrences would trigger at prose threshold but NOT at
        // HTML-code threshold (16).
        assert!(!check_loop_watchdog(phrase, &mut buf, false));
        // Continue adding occurrences to 15 — should NOT fire.
        for _ in 0..11 {
            assert!(!check_loop_watchdog(phrase, &mut buf, false));
        }
    }

    #[test]
    fn watchdog_context_hint_uses_code_threshold_after_html_prefix_rolls_out() {
        let mut buf = String::new();
        let phrase = "shared generated layout phrase for repeated panels\n";
        assert!(phrase.len() >= 30);
        for _ in 0..15 {
            assert!(!check_loop_watchdog_with_context(
                phrase, &mut buf, false, true
            ));
        }
        assert!(check_loop_watchdog_with_context(
            phrase, &mut buf, false, true
        ));
    }

    #[test]
    fn watchdog_prose_threshold_still_works_without_html() {
        // Same repeated phrase but WITHOUT HTML context: should fire
        // at the prose threshold (4).
        let mut buf = String::new();
        let phrase = "This is a thoroughly detailed explanation.\n";
        assert!(phrase.len() >= 30);
        for _ in 0..3 {
            assert!(!check_loop_watchdog(phrase, &mut buf, false));
        }
        // 4th occurrence fires at prose threshold.
        assert!(check_loop_watchdog(phrase, &mut buf, false));
    }
}
