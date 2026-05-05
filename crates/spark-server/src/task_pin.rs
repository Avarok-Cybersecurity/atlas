// SPDX-License-Identifier: AGPL-3.0-only

//! Verbatim original-task re-anchoring.
//!
//! When an agent enters a loop or hits repeated tool failures, the
//! most reliable training-free intervention (per Anthropic's harness
//! design — anthropic.com/engineering/effective-harnesses-for-long-
//! running-agents — and AgentDebug arXiv:2509.25370) is to **re-
//! introduce the user's original ask verbatim** into the prompt
//! immediately before the next assistant turn. This combats two
//! related failure modes:
//!
//! - **Task amnesia / goal drift** (HORIZON arXiv:2604.11978): after
//!   a long sequence of tool failures, the model loses its grip on
//!   the user's request and pivots to "what would you like me to
//!   work on?" — Atlas observed this in claude-export.txt 2026-04-25
//!   after 4× cargo init failures.
//! - **Confirmation-bias / mode-collapse loops** (MAR arXiv:2512.20845):
//!   the model produces near-identical retries because its "next
//!   step" prediction is dominated by its own recent (failing)
//!   output. Re-injecting the original goal shifts attention back to
//!   the actual task.
//!
//! This module is intentionally **stateless**: the original user
//! request is always present in the request's `messages` array (every
//! agentic client preserves full history), so we just walk the array
//! and pluck the first user-role message. No `conversation_id`
//! tracking, no per-session state, no eviction logic.

/// Extract the verbatim text of the FIRST user-role message in the
/// conversation. This is the closest thing to the user's "original
/// goal" without any extra framing — the first turn the user typed.
///
/// Returns `None` when the messages array contains no user message,
/// or when the first user message is empty / whitespace-only (which
/// happens for opaque continuation requests).
pub fn extract_original_goal<'a, M, F>(messages: &'a [M], get_role_and_text: F) -> Option<&'a str>
where
    F: Fn(&'a M) -> (&'a str, &'a str),
{
    for m in messages {
        let (role, text) = get_role_and_text(m);
        if role == "user" && !text.trim().is_empty() {
            return Some(text);
        }
    }
    None
}

/// Build the system-reminder block that re-anchors the agent on the
/// user's original goal. Wording is generic — does NOT mention any
/// tool names or parser shapes — so it's portable across clients
/// (Claude Code, opencode, Cline, etc.) and across model families.
///
/// `n_failures` is rendered into the message so the model sees how
/// long the loop has run; this empirically helps the model recognise
/// "I'm stuck" vs "I'm progressing slowly".
///
/// Truncates `goal` at `MAX_GOAL_QUOTE_BYTES` to bound the size of
/// the reminder when the original ask is itself a giant pasted
/// document.
pub fn build_reminder(goal: &str, n_failures: usize) -> String {
    const MAX_GOAL_QUOTE_BYTES: usize = 1024;
    let quote = if goal.len() > MAX_GOAL_QUOTE_BYTES {
        let cut = goal
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= MAX_GOAL_QUOTE_BYTES)
            .unwrap_or(MAX_GOAL_QUOTE_BYTES);
        format!("{}…", goal[..cut].trim_end())
    } else {
        goal.to_string()
    };
    format!(
        "\n\n<system-reminder>\nYou have had {n_failures} consecutive failed or repeated \
         tool calls in this session. The user's ORIGINAL request was:\n\n«{quote}»\n\n\
         Do not abandon this task. Either: (a) try a fundamentally different approach \
         (different tool, different command-line args, or accomplishing the goal \
         without that tool), or (b) report the SPECIFIC blocker concisely and what \
         you would need to proceed. Do not regenerate work that already exists; do \
         not retry an identical call.\n</system-reminder>"
    )
}

/// Decide whether to inject the goal reminder. Caller passes the
/// loop-detector verdict and the consecutive-tool-error count; this
/// function encodes the policy. Kept tiny and pure so future tuning
/// happens in one place.
pub fn should_inject(loop_state_is_active: bool, consecutive_tool_errors: u32) -> bool {
    loop_state_is_active || consecutive_tool_errors >= 3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(role: &'static str, text: &'static str) -> (String, String) {
        (role.to_string(), text.to_string())
    }

    #[test]
    fn extract_original_goal_returns_first_user_message() {
        let msgs = vec![
            pair("system", "you are helpful"),
            pair("user", "build a rust axum server"),
            pair("assistant", "I'll do that"),
            pair("user", "actually never mind"),
        ];
        let got = extract_original_goal(&msgs, |m| (m.0.as_str(), m.1.as_str()));
        assert_eq!(got, Some("build a rust axum server"));
    }

    #[test]
    fn extract_original_goal_skips_empty_user_messages() {
        let msgs = vec![pair("user", "   \n  "), pair("user", "the real ask")];
        let got = extract_original_goal(&msgs, |m| (m.0.as_str(), m.1.as_str()));
        assert_eq!(got, Some("the real ask"));
    }

    #[test]
    fn extract_original_goal_returns_none_when_no_user_message() {
        let msgs = vec![
            pair("system", "you are helpful"),
            pair("assistant", "hello"),
        ];
        let got = extract_original_goal(&msgs, |m| (m.0.as_str(), m.1.as_str()));
        assert_eq!(got, None);
    }

    #[test]
    fn build_reminder_includes_goal_verbatim_and_count() {
        let r = build_reminder("create an axum server", 4);
        assert!(
            r.contains("create an axum server"),
            "must quote verbatim: {r}"
        );
        assert!(
            r.contains("4 consecutive"),
            "must mention failure count: {r}"
        );
        assert!(r.contains("system-reminder"));
    }

    #[test]
    fn build_reminder_truncates_oversized_goal() {
        let long = "a".repeat(2000);
        let r = build_reminder(&long, 3);
        assert!(
            r.len() < 2000,
            "reminder body must cap at MAX_GOAL_QUOTE_BYTES"
        );
        assert!(r.contains("…"), "truncated marker present: {}", &r[..200]);
    }

    #[test]
    fn should_inject_fires_on_loop_active_or_3_failures() {
        assert!(should_inject(true, 0), "loop alone");
        assert!(should_inject(false, 3), "3 failures alone");
        assert!(should_inject(true, 5), "both");
        assert!(!should_inject(false, 0), "neither");
        assert!(!should_inject(false, 2), "2 failures < threshold");
    }
}
