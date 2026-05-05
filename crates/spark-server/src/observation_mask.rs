// SPDX-License-Identifier: AGPL-3.0-only

//! Stale tool-failure summarisation ("observation masking").
//!
//! Reference: Wang et al., "Observation Masking" arXiv:2508.21433
//! (Aug 2025). When the same tool failure repeats N times across a
//! conversation, every prior copy of the (often multi-KB) error
//! payload sits in context, monopolising attention and inviting the
//! model to verbatim-retry the failing call. Replacing the BODY of
//! stale failures with a single-line summary keeps the model aware
//! that the failure happened — and how many times — without burning
//! tokens on identical stack traces.
//!
//! ## Policy (kept conservative to avoid breaking valid trajectories)
//!
//! - Only target messages with role `tool` or role `user` carrying
//!   tool-result-shaped content.
//! - Only mask when the body looks like a known error signature
//!   (`Exit code N`, `command not found`, `Error:`, etc.) AND the
//!   body is at least 200 bytes (small errors don't compress).
//! - **Never mask the most-recent two error-shaped messages.** The
//!   model needs the freshest error payload to act on; only earlier
//!   duplicates get summarised.
//! - The summary preserves a one-line excerpt and the attempt index
//!   so the model can see "this exact failure has happened N times."
//!
//! Pure transformation — no extra inference call, no state.

const ERROR_SIGNATURES: &[&str] = &[
    "Exit code ",
    "command not found",
    "Error:",
    "ENOENT",
    "EISDIR",
    "EACCES",
    "Permission denied",
    "<tool_use_error>",
    "InputValidationError",
    "Traceback (most recent call last)",
];

const MIN_BODY_LEN_FOR_MASK: usize = 200;

/// Classify a single message body. Cheap substring checks — no
/// regex.
pub fn looks_like_error(body: &str) -> bool {
    if body.len() < 32 {
        return false;
    }
    ERROR_SIGNATURES.iter().any(|sig| body.contains(sig))
}

/// Compute an `(start, end)` slice indices of "fresh" error messages
/// that must NOT be masked — typically the last 2 in the conversation.
/// Returns the indices into `bodies` of messages that are recent
/// enough to leave intact.
fn fresh_error_indices(
    bodies: &[(usize, &str)],
    keep_recent: usize,
) -> std::collections::HashSet<usize> {
    let mut out = std::collections::HashSet::new();
    let total = bodies.len();
    if total <= keep_recent {
        for (idx, _) in bodies {
            out.insert(*idx);
        }
        return out;
    }
    for (idx, _) in &bodies[total - keep_recent..] {
        out.insert(*idx);
    }
    out
}

/// Extract the FIRST non-empty line of `body` — used as the summary
/// excerpt. Truncates to `max_chars` to keep the summary short.
fn first_line_excerpt(body: &str, max_chars: usize) -> String {
    let line = body
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("error")
        .trim();
    if line.chars().count() <= max_chars {
        line.to_string()
    } else {
        let cut = line
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= max_chars)
            .unwrap_or(max_chars);
        format!("{}…", line[..cut].trim_end())
    }
}

/// Build the masked envelope. Plain text, no XML, so it survives any
/// chat template renderer without re-escaping concerns.
fn envelope(attempt_idx: usize, total_attempts: usize, excerpt: &str) -> String {
    format!(
        "[stale tool failure {attempt_idx}/{total_attempts}: {excerpt} — full body elided to free attention; the most recent error is preserved verbatim below.]"
    )
}

/// Pure transformation: walks message bodies and returns the masked
/// versions. Caller-agnostic — works on any iterator of
/// `(role, body)` pairs.
///
/// `bodies_in` is `(role, body)` newest-LAST. Returns a `Vec<Option<String>>`
/// parallel to the input: `Some(new_body)` means replace, `None`
/// means leave alone.
pub fn compute_masking(
    bodies_in: &[(&str, &str)],
    keep_recent_errors: usize,
) -> Vec<Option<String>> {
    // Index error-shaped tool/user bodies in document order.
    let candidates: Vec<(usize, &str)> = bodies_in
        .iter()
        .enumerate()
        .filter(|(_, (role, body))| {
            (*role == "tool" || *role == "user")
                && body.len() >= MIN_BODY_LEN_FOR_MASK
                && looks_like_error(body)
        })
        .map(|(i, (_, body))| (i, *body))
        .collect();

    if candidates.len() < 3 {
        // Nothing to compress — < 3 errors total means at most 1
        // duplicate, not worth masking.
        return vec![None; bodies_in.len()];
    }

    let fresh = fresh_error_indices(&candidates, keep_recent_errors);
    let total = candidates.len();
    let mut out = vec![None; bodies_in.len()];
    for (attempt_idx, (msg_idx, body)) in candidates.iter().enumerate() {
        if fresh.contains(msg_idx) {
            continue;
        }
        let excerpt = first_line_excerpt(body, 80);
        out[*msg_idx] = Some(envelope(attempt_idx + 1, total, &excerpt));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_like_error_recognises_common_signatures() {
        assert!(looks_like_error(
            "Exit code 127\n/bin/bash: line 1: cargo: command not found"
        ));
        assert!(looks_like_error(
            "Error: ENOENT: no such file or directory, open '/x'"
        ));
        assert!(looks_like_error(
            "Traceback (most recent call last):\n  File...\nKeyError: 'x'"
        ));
        assert!(!looks_like_error("File written successfully at /tmp/x.txt"));
        assert!(!looks_like_error("hi"));
    }

    #[test]
    fn no_masking_when_fewer_than_3_errors() {
        let body = "Exit code 127\n/bin/bash: line 1: cargo: command not found\n".repeat(10);
        let msgs = vec![
            ("user", "build something"),
            ("assistant", "ok"),
            ("tool", body.as_str()),
            ("assistant", "trying again"),
            ("tool", body.as_str()),
        ];
        let mask = compute_masking(&msgs, 2);
        assert!(mask.iter().all(|m| m.is_none()), "2 errors → no mask");
    }

    #[test]
    fn fresh_two_errors_left_intact_older_masked() {
        let body = format!(
            "{}{}",
            "Exit code 127\n/bin/bash: line 1: cargo: command not found\n",
            "x".repeat(300)
        );
        let msgs = vec![
            ("user", "build something"), // 0
            ("assistant", "trying"),     // 1
            ("tool", body.as_str()),     // 2 — error, oldest
            ("assistant", "again"),      // 3
            ("tool", body.as_str()),     // 4 — error
            ("assistant", "again"),      // 5
            ("tool", body.as_str()),     // 6 — error, FRESH
            ("assistant", "again"),      // 7
            ("tool", body.as_str()),     // 8 — error, FRESHEST
        ];
        let mask = compute_masking(&msgs, 2);
        assert!(
            mask[2].is_some(),
            "oldest error must be masked: got {:?}",
            mask[2]
        );
        assert!(
            mask[4].is_some(),
            "second-oldest error must be masked: got {:?}",
            mask[4]
        );
        assert!(mask[6].is_none(), "fresh error must stay verbatim");
        assert!(mask[8].is_none(), "freshest error must stay verbatim");
        let masked = mask[2].as_ref().unwrap();
        assert!(masked.contains("stale tool failure"));
        assert!(
            masked.contains("Exit code 127") || masked.contains("cargo"),
            "summary must include excerpt: {masked}"
        );
    }

    #[test]
    fn small_error_bodies_are_not_masked() {
        let small = "Error: x";
        let msgs = vec![
            ("tool", small),
            ("tool", small),
            ("tool", small),
            ("tool", small),
        ];
        let mask = compute_masking(&msgs, 2);
        assert!(
            mask.iter().all(|m| m.is_none()),
            "<200B errors not worth masking"
        );
    }

    #[test]
    fn non_error_tool_results_are_never_masked() {
        let success = "File written successfully at /tmp/x.txt".repeat(10);
        let msgs = vec![
            ("tool", success.as_str()),
            ("tool", success.as_str()),
            ("tool", success.as_str()),
            ("tool", success.as_str()),
        ];
        let mask = compute_masking(&msgs, 2);
        assert!(
            mask.iter().all(|m| m.is_none()),
            "successful tool results must never be masked"
        );
    }

    #[test]
    fn assistant_messages_never_masked_even_if_error_shaped() {
        let body = format!("{}{}", "Error: oh no\n", "x".repeat(300));
        let msgs = vec![
            ("assistant", body.as_str()),
            ("assistant", body.as_str()),
            ("assistant", body.as_str()),
            ("assistant", body.as_str()),
        ];
        let mask = compute_masking(&msgs, 2);
        assert!(
            mask.iter().all(|m| m.is_none()),
            "assistant messages are not tool results"
        );
    }
}
