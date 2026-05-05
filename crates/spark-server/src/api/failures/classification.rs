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
    F29EnvironmentFact, F39FailureCache, F39PermanentFailureMatch, F49DuplicateWrite,
    append_f7_reminder_to_last_user, build_f7_stall_reminder, bump_f12_tool_call_count,
    check_loop_watchdog, collect_f7_stall_buckets, f28_text_looks_like_error,
    f29_extract_binary_from_error_line, f29_extract_environment_facts,
    f29_inject_environment_facts, f31_inject_hard_refusal, f32_reposition_failed_tool_result,
    f39_build_circuit_breaker_banner, f39_build_failure_cache, f39_class_label,
    f39_detect_recent_retries, f39_extract_binary_name, f44_check_permanent_failure,
    f49_build_banner, f49_detect_duplicate_writes, f49_extract_write_path_and_content,
    f50_append_original_error, f60_disable_mtp_for_request, flush_content_sanitizer,
    prepend_reminder_to_system, recent_message_is_tool_error,
    strip_xml_leaks_from_assistant_content,
};

// Re-export sibling helpers via crate::api::* for short paths.
use super::super::inference_types::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F37FailureClass {
    BinaryMissing,
    AlreadyExists,
    PermissionDenied,
    NotFound,
    InvalidArgument,
    StallGuard,
}

pub fn f37_classify_failure(t: &str) -> Option<F37FailureClass> {
    if t.contains("[atlas-stall-guard]") || t.contains("[atlas-permanent-failure]") {
        return Some(F37FailureClass::StallGuard);
    }
    if t.contains("command not found") || t.contains(": not found") || t.contains("Exit code 127") {
        return Some(F37FailureClass::BinaryMissing);
    }
    if t.contains("already exists")
        || t.contains("cannot be run on existing")
        || t.contains("destination") && t.contains("exists")
    {
        return Some(F37FailureClass::AlreadyExists);
    }
    if t.contains("Permission denied") || t.contains("EACCES") {
        return Some(F37FailureClass::PermissionDenied);
    }
    if t.contains("No such file or directory")
        || t.contains("ENOENT")
        || t.contains("cannot access")
    {
        return Some(F37FailureClass::NotFound);
    }
    if t.contains("TypeError")
        || t.contains("ERR_INVALID_ARG_VALUE")
        || t.contains("before overwriting it")
    {
        return Some(F37FailureClass::InvalidArgument);
    }
    None
}

// ── F23 (2026-04-26): per-conversation progress tracker ──
//
// Beyond F7's `(tool, primary_arg)` bucketing and F11's per-response
// hash dedup, this third axis tracks whether tool calls are
// MAKING USEFUL PROGRESS. Per A3 research: no upstream system
// (vLLM, SGLang, TGI, TRT-LLM, Anthropic Agent SDK, Claude Code,
// Gemini CLI) does this — Atlas is first.
//
// Score each `tool` role message in the conversation:
//   - is_error / "command not found" / "Exit code"  → -1
//   - content hash matches a prior tool result      → -1
//   - all empty / no-op (e.g. mkdir on existing dir)→  0
//   - otherwise (novel result)                       → +1
//
// At `attempts >= 6 && score <= 0` → append warn reminder.
// At `attempts >= 9 && score <= 0` → append refuse reminder.
//
// Configurable via env: ATLAS_F23_WARN_ATTEMPTS, ATLAS_F23_REFUSE_ATTEMPTS.

// F46 (2026-04-26): refuse threshold lowered 9 → 6 after live cc-
// session-fix33 showed F31 firing at attempts=14 with 3 more tool
// calls emitted afterward. The original 9-cap was based on
// Anthropic's pre-regression default; for this Qwen3.5 deployment,
// once attempts >= 6 with score <= 0 the conversation is well
// past recovery and structural escalation is appropriate.
//
// Also bumped warn 6 → 4 so warn lands earlier than refuse and
// gives the model a softer first signal.
const F23_WARN_ATTEMPTS_DEFAULT: u32 = 4;
const F23_REFUSE_ATTEMPTS_DEFAULT: u32 = 6;

#[derive(Debug, Clone, Copy)]
pub struct F23ProgressMetrics {
    pub score: i32,
    pub attempts: u32,
}

pub fn f23_normalize_and_hash(content: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut h = DefaultHasher::new();
    // Cheap normalisation: strip trailing whitespace per line, strip
    // ANSI sequences (rare in tool output), collapse internal runs of
    // whitespace. Skip very-short content (<8 chars) — too noisy.
    let cleaned: String = content
        .lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    h.write(cleaned.as_bytes());
    h.finish()
}

pub fn f23_score_progress(messages: &[crate::openai::IncomingMessage]) -> F23ProgressMetrics {
    let mut score: i32 = 0;
    let mut attempts: u32 = 0;
    let mut seen_hashes: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    // F40 (2026-04-26): also track (tool, primary_arg) collisions
    // on the ASSISTANT side. A4 audit found that 3 successful
    // Cargo.toml writes scored +3 because each had unique result
    // text — masking the fact that they were the same call
    // repeating fruitlessly. Penalise the call-side collision so
    // "model writes same path 3 times" is correctly scored as
    // stalling.
    let mut seen_call_keys: std::collections::HashMap<(String, String), u32> =
        std::collections::HashMap::new();
    // F53 (2026-04-27): success-result collision tracker. Catches
    // the cc-session-fix36 mkdir-loop class — same (tool, primary_arg)
    // succeeds repeatedly with empty/Done result. F23 result-side
    // treats those as neutral; F40 collision -1/dup is offset by
    // unrelated +1 events. F53 adds a stronger penalty when the
    // same call has been issued 3+ times AND each result is the
    // same trivial success. Tracks the call-key + last-result pair
    // and penalises -2 from the 3rd occurrence onward.
    let mut last_result_for_call: std::collections::HashMap<(String, String), u64> =
        std::collections::HashMap::new();
    let mut last_call_key_seen: Option<(String, String)> = None;
    for m in messages {
        if m.role == "assistant"
            && let Some(tcs) = &m.tool_calls
        {
            attempts = attempts.saturating_add(tcs.len() as u32);
            // F40: count call-key collisions.
            for tc in tcs {
                let name = tc.function.name.clone();
                let Some(arg) = primary_arg_for_tool(&name, &tc.function.arguments) else {
                    continue;
                };
                let key = (name.clone(), arg.clone());
                let prior = *seen_call_keys
                    .entry(key.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                if prior > 1 {
                    score -= 1;
                }
                // F53: remember the most-recent assistant tool_call
                // key so we can pair it with its result below.
                last_call_key_seen = Some(key);
            }
        }
        if m.role == "tool" {
            let content = &m.content.text;
            // 1) Explicit error markers (F6 prepends "[tool error]\n").
            if content.starts_with("[tool error]")
                || content.contains("Exit code 127")
                || content.contains("command not found")
                || content.contains("Permission denied")
                || content.contains("ENOENT")
                || content.contains("EISDIR")
            {
                score -= 1;
                last_call_key_seen = None;
                continue;
            }
            // 2) Repeated content (normalised hash).
            let h = f23_normalize_and_hash(content);
            let prior = seen_hashes.get(&h).copied().unwrap_or(0);
            seen_hashes.insert(h, prior + 1);
            // F53 (2026-04-27): for trivial successful results
            // (empty / Done / OK), pair the result with the most
            // recent assistant call key. If the same (call, result)
            // pair has been seen 3+ times, penalise -2 (above and
            // beyond F40's call-key collision penalty of -1). This
            // catches repeated "mkdir succeeds with no output"
            // patterns that other detectors silently neutral on.
            let stripped = content.trim();
            let trivial_success = stripped.is_empty() || stripped == "Done" || stripped == "OK";
            if trivial_success {
                if let Some(call_key) = last_call_key_seen.take() {
                    let prev = last_result_for_call.get(&call_key).copied();
                    last_result_for_call.insert(call_key.clone(), h);
                    if let Some(p) = prev
                        && p == h
                    {
                        // same call, same trivial result, recurring →
                        // -2 from the 3rd occurrence (the 1st recur is the 2nd
                        // call; the 2nd recur is the 3rd — start penalising at
                        // the 2nd recur to align with "3+ occurrences").
                        let count = seen_call_keys.get(&call_key).copied().unwrap_or(1);
                        if count >= 3 {
                            score -= 2;
                        }
                    }
                }
                continue; // 0 from baseline; F53 already adjusted score
            }
            if prior > 0 {
                score -= 1;
                last_call_key_seen = None;
                continue;
            }
            // 3) Novel useful result.
            score += 1;
            last_call_key_seen = None;
        }
    }
    F23ProgressMetrics { score, attempts }
}

pub fn f23_warn_threshold() -> u32 {
    std::env::var("ATLAS_F23_WARN_ATTEMPTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(F23_WARN_ATTEMPTS_DEFAULT)
}

pub fn f23_refuse_threshold() -> u32 {
    std::env::var("ATLAS_F23_REFUSE_ATTEMPTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(F23_REFUSE_ATTEMPTS_DEFAULT)
}

pub fn f23_build_reminder(metrics: F23ProgressMetrics) -> Option<String> {
    let warn = f23_warn_threshold();
    let refuse = f23_refuse_threshold();
    if metrics.attempts >= refuse && metrics.score <= 0 {
        Some(format!(
            "\n\n<system-reminder>\n\
             [F23 STALL] You have called {} tools without making progress \
             (progress_score={}). STOP. Do NOT call any more tools. Reply \
             to the user with a plain-text explanation of what is blocking \
             and what you need from them. Subsequent tool calls will be \
             refused.\n\
             </system-reminder>",
            metrics.attempts, metrics.score
        ))
    } else if metrics.attempts >= warn && metrics.score <= 0 {
        Some(format!(
            "\n\n<system-reminder>\n\
             [F23 STALL] You have called {} tools without making progress \
             (progress_score={}). The retries are not advancing the task. \
             Stop and explain to the user what is blocking. Do not retry \
             the same approach.\n\
             </system-reminder>",
            metrics.attempts, metrics.score
        ))
    } else {
        None
    }
}
