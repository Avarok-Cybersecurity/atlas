// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

// Sibling-cluster items hoisted from the original `api.rs`. These uses
// give every sub-file access to helpers that the un-split file took for
// granted via single-module visibility.
use super::super::chat::chat_completions_inner;
use super::super::compact::{
    compact_messages, openai_error_response, openai_error_response_with_param,
};
use super::super::completions::not_supported;
use super::super::inference_impl::{
    extract_thinking, strip_stop_sequences, tokenize_stop_sequences,
};
use super::super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};
use super::super::strip::strip_thinking_tags;
use super::{
    F23ProgressMetrics, F37FailureClass, F39FailureCache, F39PermanentFailureMatch,
    F49DuplicateWrite, bump_f12_tool_call_count, check_loop_watchdog, f23_build_reminder,
    f23_normalize_and_hash, f23_refuse_threshold, f23_score_progress, f23_warn_threshold,
    f31_inject_hard_refusal, f32_reposition_failed_tool_result, f37_classify_failure,
    f39_build_circuit_breaker_banner, f39_build_failure_cache, f39_class_label,
    f39_detect_recent_retries, f39_extract_binary_name, f44_check_permanent_failure,
    f49_build_banner, f49_detect_duplicate_writes, f49_extract_write_path_and_content,
    f50_append_original_error, f60_disable_mtp_for_request, flush_content_sanitizer,
    strip_xml_leaks_from_assistant_content,
};

// Re-export sibling helpers via crate::api::* for short paths.
use super::super::inference_types::*;

pub fn collect_f7_stall_buckets(messages: &[crate::openai::IncomingMessage]) -> F7StallBuckets {
    let mut buckets = F7StallBuckets::new();
    for m in messages {
        if m.role != "assistant" {
            continue;
        }
        let Some(tcs) = &m.tool_calls else {
            continue;
        };
        for tc in tcs {
            let name = tc.function.name.clone();
            let Some(arg) = primary_arg_for_tool(&name, &tc.function.arguments) else {
                continue;
            };
            *buckets.entry((name, arg)).or_insert(0) += 1;
        }
    }
    buckets
}

/// Build the system-reminder text covering all buckets at or above the
/// warn threshold. Returns `None` if no bucket triggers.
pub fn build_f7_stall_reminder(buckets: &F7StallBuckets) -> Option<String> {
    let mut warn_lines: Vec<String> = buckets
        .iter()
        .filter(|(_, c)| **c >= F7_STALL_WARN_THRESHOLD)
        .map(|((name, arg), c)| format!("- {}({}) × {}", name, arg, c))
        .collect();
    if warn_lines.is_empty() {
        return None;
    }
    warn_lines.sort();
    let any_at_refuse = buckets.values().any(|&c| c >= F7_STALL_REFUSE_THRESHOLD);
    let body = warn_lines.join("\n");
    let directive = if any_at_refuse {
        "STOP retrying. Respond to the user with a plain text message explaining what is blocking and what you need from them. Do NOT call any tool again — your next response must contain text only."
    } else {
        "The retries are not making progress. Stop and explain to the user what is blocking and what you need from them. Do not call these tools again with the same arguments."
    };
    Some(format!(
        "\n\n<system-reminder>\nYou have already issued the following tool calls multiple times in this conversation:\n{body}\n{directive}\n</system-reminder>"
    ))
}

/// Append the reminder text to the last user/tool message. If neither
/// exists in the conversation, push a synthetic system message.
pub fn append_f7_reminder_to_last_user(
    messages: &mut Vec<crate::openai::IncomingMessage>,
    reminder: &str,
) {
    for m in messages.iter_mut().rev() {
        if m.role == "user" || m.role == "tool" {
            m.content.text.push_str(reminder);
            return;
        }
    }
    // No user/tool message — push a synthetic system note. Trim
    // leading whitespace so the synthetic message doesn't start with
    // blank lines.
    messages.push(crate::openai::IncomingMessage::synthetic_system(
        reminder.trim_start().to_string(),
    ));
}

/// F30 (2026-04-26): prepend a reminder to the system message at
/// position 0. Per L2 audit + arXiv:2508.15815: reminders inside
/// `<tool_response>` envelopes are RL-trained as ambient observation
/// data and ignored. Prepending to system places the signal at the
/// highest-attention position. Wrapped in `<atlas_runtime_notice>`
/// so vendor-trained position-sensitive templates aren't confused
/// by the injected content.
///
/// Idempotent: if a previous notice exists, it's replaced with the
/// fresh content. If no system message exists, a synthetic one is
/// inserted at position 0.
pub fn prepend_reminder_to_system(
    messages: &mut Vec<crate::openai::IncomingMessage>,
    reminder: &str,
) {
    let trimmed = reminder.trim_matches(|c: char| c == '\n' || c == ' ');
    let block = format!("<atlas_runtime_notice>\n{trimmed}\n</atlas_runtime_notice>\n\n");
    let has_system_at_zero = messages.first().is_some_and(|m| m.role == "system");
    if has_system_at_zero {
        let txt = &mut messages[0].content.text;
        if let Some(start) = txt.find("<atlas_runtime_notice>")
            && let Some(end) = txt[start..].find("</atlas_runtime_notice>\n\n")
        {
            let abs_end = start + end + "</atlas_runtime_notice>\n\n".len();
            txt.replace_range(start..abs_end, &block);
            return;
        }
        let mut new_txt = block;
        new_txt.push_str(txt);
        messages[0].content.text = new_txt;
    } else {
        messages.insert(
            0,
            crate::openai::IncomingMessage::synthetic_system(block.trim_end().to_string()),
        );
    }
}

// ── F29 (2026-04-26): environment-facts auto-injection ──
//
// When Atlas observes a binary failing 3+ times with `command not
// found` / `Exit code 127` patterns, that's structurally a permanent
// fact about the environment — the binary will not become available
// mid-conversation. F29 extracts these facts from message history
// each turn and PREPENDS them to the system message, where Qwen3.5
// is RL-trained to treat content as authoritative (vs. inside
// `<tool_response>` which is treated as ambient observation data
// per ChatBug arXiv:2406.12935 / arXiv:2508.15815).
//
// Stateless: rebuilt each turn from message history. Threshold of
// 3 occurrences filters false positives (transient "binary missing
// then installed" sequences are rare in agentic workflows).

// F36 (2026-04-26): lowered from 3 → 2 after L1 audit of fix32:
// in cc-session-axum-test-22 the model retried `cargo init` after
// only ONE prior `command not found` (total 2 failures); F29 at
// threshold=3 never fired. Two observed failures of the same
// binary in agentic workflows is overwhelming evidence the binary
// is permanently unavailable — installs don't happen mid-conversation.
const F29_MIN_FAILURES_FOR_FACT: u32 = 2;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct F29EnvironmentFact {
    binary: String,
    observed_count: u32,
}

/// Extract a binary name from a `command not found` line. Returns
/// the binary if the line matches the expected error format,
/// otherwise None.
///
/// Patterns matched:
///   "X: command not found"
///   "/bin/bash: line 1: X: command not found"
///   "X: not found"
///   "bash: X: command not found"
///   "/bin/sh: 1: X: not found"
///
/// `Exit code 127` lines without an inline binary name are NOT
/// matched here — F37's `f37_failure_signature` covers those at the
/// per-tool-call level instead.
pub fn f29_extract_binary_from_error_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let suffix = ": command not found";
    let alt_suffix = ": not found";
    let idx = trimmed
        .rfind(suffix)
        .or_else(|| trimmed.rfind(alt_suffix))?;
    let prefix = &trimmed[..idx];
    // Take the LAST whitespace-or-`:`-separated token before the
    // suffix as the binary name. E.g.
    //   "/bin/bash: line 1: cargo" → "cargo"
    //   "cargo" → "cargo"
    let binary = prefix.rsplit([' ', ':', '\t']).next()?.trim();
    // Sanity: binary names are typically [A-Za-z0-9_-]+, no spaces,
    // not too long. Reject if it looks like a path or sentence.
    if binary.is_empty()
        || binary.len() > 40
        || binary.contains('/')
        || !binary
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return None;
    }
    Some(binary.to_string())
}

pub fn f29_extract_environment_facts(
    messages: &[crate::openai::IncomingMessage],
) -> Vec<F29EnvironmentFact> {
    let mut counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for m in messages {
        if m.role != "tool" {
            continue;
        }
        for line in m.content.text.lines() {
            if let Some(bin) = f29_extract_binary_from_error_line(line) {
                *counts.entry(bin).or_insert(0) += 1;
            }
        }
    }
    let mut facts: Vec<F29EnvironmentFact> = counts
        .into_iter()
        .filter_map(|(binary, observed_count)| {
            if observed_count >= F29_MIN_FAILURES_FOR_FACT {
                Some(F29EnvironmentFact {
                    binary,
                    observed_count,
                })
            } else {
                None
            }
        })
        .collect();
    facts.sort_by(|a, b| a.binary.cmp(&b.binary));
    facts
}

/// Prepend an `<environment_facts>` block to the (existing or new)
/// system message at position 0. Idempotent: if the block already
/// exists in the system message, replace it with the fresh count.
pub fn f29_inject_environment_facts(
    messages: &mut Vec<crate::openai::IncomingMessage>,
    facts: &[F29EnvironmentFact],
) {
    if facts.is_empty() {
        return;
    }
    let lines: Vec<String> = facts
        .iter()
        .map(|f| {
            format!(
                "- `{}` (returned 'command not found' {} time{})",
                f.binary,
                f.observed_count,
                if f.observed_count == 1 { "" } else { "s" }
            )
        })
        .collect();
    let block = format!(
        "<environment_facts>\nThe following commands are NOT available in this environment. Do NOT attempt to use them — the failure is permanent and retrying with cosmetic variations (mkdir prefix, cd path, flag order) will not change the outcome:\n{}\n</environment_facts>\n\n",
        lines.join("\n")
    );

    // Locate or synthesise system message at position 0.
    let has_system_at_zero = messages.first().is_some_and(|m| m.role == "system");
    if has_system_at_zero {
        // Idempotent replace: if a previous block exists, swap it.
        let txt = &mut messages[0].content.text;
        if let Some(start) = txt.find("<environment_facts>")
            && let Some(end) = txt[start..].find("</environment_facts>\n\n")
        {
            let abs_end = start + end + "</environment_facts>\n\n".len();
            txt.replace_range(start..abs_end, &block);
            return;
        }
        // No prior block — prepend.
        let mut new_txt = block;
        new_txt.push_str(txt);
        messages[0].content.text = new_txt;
    } else {
        messages.insert(
            0,
            crate::openai::IncomingMessage::synthetic_system(block.trim_end().to_string()),
        );
    }
}

// F28 (2026-04-26): test whether the most recent message in the
// conversation is a tool result that errored. Used to gate auto-
// disable-thinking and other failure-recovery behaviour. Looks at
// the LAST `role: tool` message (from any position near the tail)
// and checks for the F6 `[tool error]` prefix or the well-known
// non-retryable patterns the `NotInstalledHint` injector uses.
//
// F41 (2026-04-26): expanded patterns to cover opencode-style
// errors that the L6 audit found bypassing F28: TypeError,
// ERR_INVALID_ARG_VALUE, "cannot access", "No such file or
// directory", "already exists", "must read file ... before
// overwriting", and bare "Error:" prefix. opencode forwards raw
// node/zod/tool-runtime errors without an `[tool error]` marker
// because it routes through OpenAI direct (no Anthropic translator).
pub fn recent_message_is_tool_error(messages: &[crate::openai::IncomingMessage]) -> bool {
    for m in messages.iter().rev() {
        if m.role == "tool" {
            let t = &m.content.text;
            return f28_text_looks_like_error(t);
        }
    }
    false
}

pub fn f28_text_looks_like_error(t: &str) -> bool {
    if t.starts_with("[tool error]") || t.starts_with("Error:") {
        return true;
    }
    // F37/F41 expanded permanent-failure patterns. Order: most
    // common first for short-circuit.
    const SIGNATURES: &[&str] = &[
        "Exit code 127",
        "command not found",
        ": not found",
        "Permission denied",
        "[atlas-stall-guard]",
        "[atlas-permanent-failure]",
        // F41 (opencode coverage)
        "TypeError",
        "ERR_INVALID_ARG_VALUE",
        "cannot access",
        "No such file or directory",
        // F37 (env-state failures observed in opencode)
        "already exists",
        "cannot be run on existing",
        "before overwriting it",
        "ENOENT",
        "EISDIR",
        "EACCES",
    ];
    SIGNATURES.iter().any(|s| t.contains(s))
}
