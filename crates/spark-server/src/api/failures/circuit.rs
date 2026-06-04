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
    F23ProgressMetrics, F29EnvironmentFact, F37FailureClass, F49DuplicateWrite,
    append_f7_reminder_to_last_user, build_f7_stall_reminder, bump_f12_tool_call_count,
    check_loop_watchdog, collect_f7_stall_buckets, f23_build_reminder, f23_normalize_and_hash,
    f23_refuse_threshold, f23_score_progress, f23_warn_threshold, f28_text_looks_like_error,
    f29_extract_binary_from_error_line, f29_extract_environment_facts,
    f29_inject_environment_facts, f37_classify_failure, f49_build_banner,
    f49_detect_duplicate_writes, f49_extract_write_path_and_content, f50_append_original_error,
    flush_content_sanitizer, prepend_reminder_to_system, recent_message_is_tool_error,
    strip_xml_leaks_from_assistant_content,
};

// Re-export sibling helpers via crate::api::* for short paths.
use super::super::inference_types::*;

pub fn f32_reposition_failed_tool_result(
    messages: &mut Vec<crate::openai::IncomingMessage>,
) -> bool {
    let mut last_err_idx: Option<usize> = None;
    for (i, m) in messages.iter().enumerate().rev() {
        if m.role == "tool" {
            let t = &m.content.text;
            if t.starts_with("[tool error]")
                || t.contains("Exit code 127")
                || t.contains("command not found")
                || t.contains("[atlas-stall-guard]")
            {
                last_err_idx = Some(i);
            }
            break;
        }
    }
    let idx = match last_err_idx {
        Some(v) => v,
        None => return false,
    };
    // Need at least 2 messages after the failed tool result for the
    // gap to be worth bridging; otherwise the original IS at the
    // tail and a duplicate is wasteful.
    if messages.len().saturating_sub(idx + 1) < 2 {
        return false;
    }
    // Don't duplicate the last message if it's already a stall-guard
    // refusal (F31 already put a freshness signal at the tail).
    if let Some(last) = messages.last()
        && last.role == "tool"
        && last
            .content
            .text
            .starts_with("[tool error]\n[atlas-stall-guard]")
    {
        return false;
    }
    let failed_tool_result = messages[idx].content.text.trim();
    let reminder = format!(
        "\n\n<system-reminder>\nThe most recent failed tool result is still relevant. \
         Treat it as an observed tool failure and do not retry the same call unless \
         the approach materially changes.\n<failed_tool_result>\n{failed_tool_result}\n\
         </failed_tool_result>\n</system-reminder>"
    );
    append_f7_reminder_to_last_user(messages, &reminder);
    true
}

/// F31 (2026-04-26): when F23 trips its refuse threshold AND the
/// most recent assistant turn included tool_calls, synthesise a
/// `role: tool` message with `[atlas-stall-guard]` content anchored
/// to the first tool_use_id from that turn. Per A2/L3 + BFCL-v3
/// ablation: structured `is_error: true` payloads invoke the
/// model's RL-trained error-recovery pathway (78% recovery vs 42%
/// for soft `<system-reminder>`). The synthesised message is pushed
/// AFTER the most recent assistant turn (and any existing tool
/// results from that turn) so the model sees it as the freshest
/// observation.
///
/// Returns true when a refusal message was injected.
pub fn f31_inject_hard_refusal(
    messages: &mut Vec<crate::openai::IncomingMessage>,
    metrics: F23ProgressMetrics,
) -> bool {
    if metrics.attempts < f23_refuse_threshold() || metrics.score > 0 {
        return false;
    }
    // Find the most recent assistant turn's first tool_call id and
    // its index. We anchor the synthesised tool_result to a real id
    // so transport-level frame validators don't drop it.
    let mut anchor: Option<(usize, String)> = None;
    for (i, m) in messages.iter().enumerate().rev() {
        if m.role == "assistant" {
            if let Some(tcs) = &m.tool_calls
                && let Some(first) = tcs.first()
                && let Some(id) = first.id.as_ref()
            {
                anchor = Some((i, id.clone()));
            }
            break;
        }
    }
    let (asst_idx, tool_call_id) = match anchor {
        Some(v) => v,
        None => return false,
    };

    // Idempotent on the SAME tool_call_id: if a stall-guard message
    // is already attached to this id, don't add another. (A new
    // assistant turn produces a new tool_call_id, so the next turn
    // gets a fresh stall-guard naturally.)
    if messages.iter().any(|m| {
        m.role == "tool"
            && m.tool_call_id.as_deref() == Some(tool_call_id.as_str())
            && m.content
                .text
                .starts_with("[tool error]\n[atlas-stall-guard]")
    }) {
        return false;
    }

    // F46 (2026-04-26): count prior stall-guards to escalate the
    // refusal wording on each subsequent firing. The first banner
    // is firm; the second is severe; the third asserts that
    // continued retries will be silently dropped.
    let prior_stall_guards = messages
        .iter()
        .filter(|m| {
            m.role == "tool"
                && m.content
                    .text
                    .starts_with("[tool error]\n[atlas-stall-guard]")
        })
        .count();

    let directive = match prior_stall_guards {
        0 => {
            "STOP retrying. Reply to the user with a plain-text explanation of what is blocking and what you need from them. Do NOT call any tool again — your next response must contain text only."
        }
        1 => {
            "SECOND REFUSAL. The previous stall-guard was ignored — you continued issuing tool calls. This is not productive. Stop retrying immediately. Reply to the user in plain text with a description of what is blocking; do not call any tool, do not paste code, do not narrate next steps."
        }
        _ => {
            "REPEATED REFUSALS. Atlas has injected multiple stall-guards and you continued generating tool calls. The agentic harness is not making progress. Acknowledge to the user that the task cannot be completed in this environment and explain why. Do not emit any further tool calls."
        }
    };
    let body = format!(
        "[tool error]\n[atlas-stall-guard] This conversation has made {} tool \
         calls with no progress (score={}). Atlas has refused this tool call \
         (refusal #{}). {}",
        metrics.attempts,
        metrics.score,
        prior_stall_guards + 1,
        directive
    );
    let synth = crate::openai::IncomingMessage {
        role: "tool".to_string(),
        content: crate::openai::ParsedContent {
            text: body,
            images: Vec::new(),
        },
        tool_calls: None,
        tool_call_id: Some(tool_call_id),
        name: None,
    };

    // Insert after the assistant turn AND after any tool results
    // already attached to that turn so the synthesised message is
    // the absolute-tail tool result the model sees.
    let mut insert_at = asst_idx + 1;
    while insert_at < messages.len() && messages[insert_at].role == "tool" {
        insert_at += 1;
    }
    messages.insert(insert_at, synth);
    true
}

// ── F39 (2026-04-26): cross-turn permanent-failure circuit breaker ──
//
// Per A2 (Algorithmic Circuit Breakers — DZone, Cordum 2026):
// the mainstream pattern for stopping the documented 1-2 retry
// floor in RL-trained coder models. The literature (PALADIN
// arXiv:2509.25238, MAR arXiv:2512.20845, Reflexion arXiv:2303.11366)
// is unanimous: prompt-level dissuasion has a 2-retry floor that
// can't be removed by stronger prompts — the policy was REWARDED in
// RL for retrying with mild variation.
//
// F39 walks history once, builds a `(tool, primary_arg) ->
// permanent_failure_class` cache from past tool_results, then scans
// the LAST assistant turn's tool_calls for matches. When a match is
// found, an `[atlas-permanent-failure]` banner is prepended to the
// system message — wrapped in `<atlas_runtime_notice>` so it lands
// at messages[0]:tail (highest-attention slot per L5 audit).
//
// This is *additive* with F23/F31: F39 fires earlier (at the SECOND
// occurrence of the same call), F23 fires at 6+ attempts of any
// stalling pattern, F31 fires at 9+ of unprogressing.

#[derive(Debug, Clone)]
pub struct F39PermanentFailureMatch {
    pub tool: String,
    pub primary_arg: String,
    pub class: F37FailureClass,
    pub prior_failure_count: u32,
}

/// F39 cache: `(tool, primary_arg) → (count, class)` for direct
/// matches, plus a per-tool fallback set of "this tool's first-word
/// binaries that are known missing". The fallback catches retries
/// like `cargo init --offline` after `cargo init` failed — same
/// missing binary, different flags.
#[derive(Debug, Default)]
pub struct F39FailureCache {
    pub direct: std::collections::HashMap<(String, String), (u32, F37FailureClass)>,
    /// Tool-name → set of first-word binaries known to be missing.
    /// Only populated for `BinaryMissing` class (where ANY future
    /// invocation of that binary will fail).
    pub missing_bins_by_tool:
        std::collections::HashMap<String, std::collections::HashMap<String, u32>>,
}

/// Extract the binary name (first word) from a Bash command, after
/// stripping any leading `cd ... &&` boilerplate. For non-Bash
/// tools, returns None. F51: case-insensitive tool match.
pub fn f39_extract_binary_name(tool: &str, primary_arg: &str) -> Option<String> {
    if classify_tool(tool) != ToolKind::Bash {
        return None;
    }
    let first_word = primary_arg.split_whitespace().next()?;
    if first_word.is_empty() {
        return None;
    }
    Some(first_word.to_string())
}

pub fn f39_build_failure_cache(messages: &[crate::openai::IncomingMessage]) -> F39FailureCache {
    let mut cache = F39FailureCache::default();
    let mut pending_calls: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();
    for m in messages {
        if m.role == "assistant" {
            if let Some(tcs) = &m.tool_calls {
                for tc in tcs {
                    let id = tc.id.clone().unwrap_or_default();
                    if id.is_empty() {
                        continue;
                    }
                    let name = tc.function.name.clone();
                    let Some(arg) = primary_arg_for_tool(&name, &tc.function.arguments) else {
                        continue;
                    };
                    pending_calls.insert(id, (name, arg));
                }
            }
        } else if m.role == "tool" {
            let Some(id) = m.tool_call_id.as_ref() else {
                continue;
            };
            let Some((name, arg)) = pending_calls.remove(id) else {
                continue;
            };
            if let Some(class) = f37_classify_failure(&m.content.text) {
                let entry = cache
                    .direct
                    .entry((name.clone(), arg.clone()))
                    .or_insert((0, class));
                entry.0 += 1;
                entry.1 = class;
                // F39 fallback: when the binary is missing, ALL
                // future invocations with the same first-word will
                // fail. Cache the binary name separately.
                if class == F37FailureClass::BinaryMissing
                    && let Some(bin) = f39_extract_binary_name(&name, &arg)
                {
                    let bm = cache.missing_bins_by_tool.entry(name).or_default();
                    *bm.entry(bin).or_insert(0) += 1;
                }
            }
        }
    }
    cache
}

/// Detect tool_calls in the most recent assistant turn that match
/// a (tool, primary_arg) in the failure cache, OR (for Bash) match
/// a previously-failed binary by first-word. Returns one match per
/// failing key (deduplicated within the turn).
pub fn f39_detect_recent_retries(
    messages: &[crate::openai::IncomingMessage],
) -> Vec<F39PermanentFailureMatch> {
    let cache = f39_build_failure_cache(messages);
    if cache.direct.is_empty() && cache.missing_bins_by_tool.is_empty() {
        return Vec::new();
    }
    let mut last_asst: Option<&crate::openai::IncomingMessage> = None;
    for m in messages.iter().rev() {
        if m.role == "assistant" {
            last_asst = Some(m);
            break;
        }
    }
    let Some(asst) = last_asst else {
        return Vec::new();
    };
    let Some(tcs) = &asst.tool_calls else {
        return Vec::new();
    };
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    let mut matches: Vec<F39PermanentFailureMatch> = Vec::new();
    for tc in tcs {
        let name = tc.function.name.clone();
        let Some(arg) = primary_arg_for_tool(&name, &tc.function.arguments) else {
            continue;
        };
        let key = (name.clone(), arg.clone());
        if !seen.insert(key.clone()) {
            continue;
        }
        // 1) Direct match against (tool, primary_arg).
        if let Some((count, class)) = cache.direct.get(&key)
            && *count >= 1
        {
            matches.push(F39PermanentFailureMatch {
                tool: name.clone(),
                primary_arg: arg.clone(),
                class: *class,
                prior_failure_count: *count,
            });
            continue;
        }
        // 2) Fallback for Bash: first-word binary lookup. Catches
        //    `cargo init --offline` retry after `cargo init` failed
        //    with `command not found` — same missing binary.
        if let Some(bin) = f39_extract_binary_name(&name, &arg)
            && let Some(bm) = cache.missing_bins_by_tool.get(&name)
            && let Some(count) = bm.get(&bin)
        {
            matches.push(F39PermanentFailureMatch {
                tool: name,
                primary_arg: arg,
                class: F37FailureClass::BinaryMissing,
                prior_failure_count: *count,
            });
        }
    }
    matches
}

pub fn f39_class_label(class: F37FailureClass) -> &'static str {
    match class {
        F37FailureClass::BinaryMissing => "binary not installed (command not found / exit 127)",
        F37FailureClass::AlreadyExists => "destination already exists",
        F37FailureClass::PermissionDenied => "permission denied",
        F37FailureClass::NotFound => "path/file not found",
        F37FailureClass::InvalidArgument => "invalid argument or environment-state error",
        F37FailureClass::StallGuard => "Atlas stall-guard refused this call",
    }
}

pub fn f39_build_circuit_breaker_banner(matches: &[F39PermanentFailureMatch]) -> String {
    let lines: Vec<String> = matches
        .iter()
        .map(|m| {
            format!(
                "- {}({}) — {} (failed {} time{} previously)",
                m.tool,
                m.primary_arg,
                f39_class_label(m.class),
                m.prior_failure_count,
                if m.prior_failure_count == 1 { "" } else { "s" }
            )
        })
        .collect();
    format!(
        "<atlas_circuit_breaker>\n\
         CRITICAL: You are repeating tool calls that have already FAILED with PERMANENT errors. The retries below cannot succeed in this environment — the failure is structural (binary missing, file/path absent, permission denied, etc.) and will not change between attempts:\n\
         {}\n\n\
         STOP retrying these calls. Pick ONE of these CONCRETE next actions:\n\
         (a) If the failed tool was a missing binary (cargo, npm, etc.): use the Write tool to create the project files MANUALLY — write the Cargo.toml, src/main.rs, and any other source files directly with their full contents. Do NOT call the missing binary again.\n\
         (b) If the failed tool used a wrong path or argument: try Bash with a SUBSTANTIVELY different command (different directory, different flags, completely different approach). Do NOT vary cosmetically.\n\
         (c) If neither (a) nor (b) is feasible: reply to the user in PLAIN TEXT (no tool call at all) explaining specifically what is blocking and what dependency or permission they need to provide.\n\
         Do NOT issue the same failing tool call again. Do NOT emit a generic \"please clarify your request\" question — the user already gave the request. Pick (a), (b), or (c) and execute it.\n\
         </atlas_circuit_breaker>",
        lines.join("\n")
    )
}

/// F44 (2026-04-27): streaming-level lookup against the F39 cache.
/// Returns true when the in-progress tool_call would be a retry of a
/// known permanent failure. Reuses `primary_arg_for_tool` (file_path
/// for Write/Edit/Read, `extract_bash_final_action` for Bash) and
/// the F39 cache's first-word-binary fallback for Bash.
pub fn f44_check_permanent_failure(cache: &F39FailureCache, tool: &str, args_json: &str) -> bool {
    let Some(primary_arg) = primary_arg_for_tool(tool, args_json) else {
        tracing::debug!(tool = %tool, "F44/F55: primary_arg_for_tool returned None");
        return false;
    };
    let key = (tool.to_string(), primary_arg.clone());
    if cache.direct.contains_key(&key) {
        tracing::info!(
            tool = %tool,
            primary_arg = %primary_arg,
            "F44/F55: direct cache hit — suppressing tool_call"
        );
        return true;
    }
    if let Some(bin) = f39_extract_binary_name(tool, &primary_arg)
        && let Some(missing) = cache.missing_bins_by_tool.get(tool)
        && missing.contains_key(&bin)
    {
        tracing::info!(
            tool = %tool,
            bin = %bin,
            "F44/F55: bin-fallback cache hit — suppressing tool_call"
        );
        return true;
    }
    tracing::debug!(
        tool = %tool,
        primary_arg = %primary_arg,
        cache_direct_size = cache.direct.len(),
        "F44/F55: no cache match"
    );
    false
}
