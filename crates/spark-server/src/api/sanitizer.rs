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
use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::failures::{
    F23ProgressMetrics, F29EnvironmentFact, F37FailureClass, F39FailureCache,
    F39PermanentFailureMatch, F49DuplicateWrite, append_f7_reminder_to_last_user,
    build_f7_stall_reminder, bump_f12_tool_call_count, check_loop_watchdog,
    collect_f7_stall_buckets, f23_build_reminder, f23_normalize_and_hash, f23_refuse_threshold,
    f23_score_progress, f23_warn_threshold, f28_text_looks_like_error,
    f29_extract_binary_from_error_line, f29_extract_environment_facts,
    f29_inject_environment_facts, f31_inject_hard_refusal, f32_reposition_failed_tool_result,
    f37_classify_failure, f39_build_circuit_breaker_banner, f39_build_failure_cache,
    f39_class_label, f39_detect_recent_retries, f39_extract_binary_name,
    f44_check_permanent_failure, f49_build_banner, f49_detect_duplicate_writes,
    f49_extract_write_path_and_content, f50_append_original_error, f60_disable_mtp_for_request,
    flush_content_sanitizer, prepend_reminder_to_system, recent_message_is_tool_error,
    strip_xml_leaks_from_assistant_content,
};
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::strip::strip_thinking_tags;

// Re-export sibling helpers via crate::api::* for short paths.
use super::inference_types::*;

pub fn sanitize_content_chunk(
    text: &str,
    tag_scan_buf: &mut String,
    suppressing_param_leak: &mut bool,
    inside_envelope: &mut bool,
    markers: &tool_parser::LeakMarkers,
) -> String {
    // Fast-path: parser opted out of sanitization (default for Hermes,
    // Gemma4, Mistral, BareJson). Pass the text straight through without
    // buffering, so no tail-retention latency penalty for those deployments.
    if markers.orphan_open.is_empty() && markers.envelope_open.is_empty() {
        return text.to_string();
    }
    // Keep enough trailing bytes buffered that a partial tag straddling
    // a chunk boundary can fuse with the next chunk.
    let tag_max: usize = markers
        .orphan_open
        .iter()
        .chain(markers.close.iter())
        .chain(markers.envelope_open.iter())
        .chain(markers.envelope_close.iter())
        .map(|t| t.len())
        .max()
        .unwrap_or(0);

    tag_scan_buf.push_str(text);
    let mut out = String::new();
    loop {
        if *suppressing_param_leak {
            let earliest = markers
                .close
                .iter()
                .filter_map(|t| tag_scan_buf.find(t).map(|p| (p, t.len())))
                .min_by_key(|(p, _)| *p);
            match earliest {
                Some((pos, len)) => {
                    tag_scan_buf.drain(..pos + len);
                    *suppressing_param_leak = false;
                }
                None => {
                    if tag_scan_buf.len() > tag_max.saturating_sub(1) {
                        let keep = tag_max.saturating_sub(1);
                        let drop_to = tag_scan_buf.len() - keep;
                        let cut = tag_scan_buf
                            .char_indices()
                            .map(|(i, _)| i)
                            .take_while(|&i| i <= drop_to)
                            .last()
                            .unwrap_or(0);
                        tag_scan_buf.drain(..cut);
                    }
                    break;
                }
            }
            continue;
        }

        // F73 (2026-04-29): match envelope markers first so an
        // envelope_open (e.g. `<minimax:tool_call>`) takes
        // precedence over a stray-looking inner `<invoke ...>`. Inside
        // an envelope, orphan_open is suppressed-skip — the inner
        // tags are part of the legitimate (if mangled) tool call.
        let earliest_env_open = markers
            .envelope_open
            .iter()
            .filter_map(|t| tag_scan_buf.find(t).map(|p| (p, t.len())))
            .min_by_key(|(p, _)| *p);
        let earliest_env_close = markers
            .envelope_close
            .iter()
            .filter_map(|t| tag_scan_buf.find(t).map(|p| (p, t.len())))
            .min_by_key(|(p, _)| *p);
        // Inside an envelope, skip BOTH orphan_open and orphan_close
        // matching — the inner `<invoke>...<parameter>...</parameter>
        // </invoke>` content is legitimate and must pass through
        // unchanged. Orphan close tags only get dropped when they
        // appear outside any envelope (true stray fragments).
        let (earliest_open, earliest_close) = if *inside_envelope {
            (None, None)
        } else {
            (
                markers
                    .orphan_open
                    .iter()
                    .filter_map(|t| tag_scan_buf.find(t).map(|p| (p, t.len())))
                    .min_by_key(|(p, _)| *p),
                markers
                    .close
                    .iter()
                    .filter_map(|t| tag_scan_buf.find(t).map(|p| (p, t.len())))
                    .min_by_key(|(p, _)| *p),
            )
        };
        // Action variants: (pos, len, kind) where kind selects the
        // state transition. Tie-break: envelope > orphan > close at
        // the same position so an envelope_open consumes its bytes
        // before any orphan-suppression triggers.
        #[derive(Copy, Clone)]
        enum ActKind {
            EnvelopeOpen,
            EnvelopeClose,
            OrphanOpen,
            OrphanClose,
        }
        let mut best: Option<(usize, usize, ActKind)> = None;
        let consider = |cand: Option<(usize, usize)>,
                        kind: ActKind,
                        best: &mut Option<(usize, usize, ActKind)>| {
            if let Some((p, l)) = cand {
                match best {
                    None => *best = Some((p, l, kind)),
                    Some((bp, _, _)) if p < *bp => *best = Some((p, l, kind)),
                    _ => {}
                }
            }
        };
        consider(earliest_env_open, ActKind::EnvelopeOpen, &mut best);
        consider(earliest_env_close, ActKind::EnvelopeClose, &mut best);
        consider(earliest_open, ActKind::OrphanOpen, &mut best);
        consider(earliest_close, ActKind::OrphanClose, &mut best);

        match best {
            Some((pos, tag_len, kind)) => {
                let before: String = tag_scan_buf.drain(..pos).collect();
                out.push_str(&before);
                match kind {
                    ActKind::EnvelopeOpen => {
                        // Emit the envelope_open bytes — they're
                        // legitimate content the user should see —
                        // and switch state.
                        let env_bytes: String = tag_scan_buf.drain(..tag_len).collect();
                        out.push_str(&env_bytes);
                        *inside_envelope = true;
                    }
                    ActKind::EnvelopeClose => {
                        let env_bytes: String = tag_scan_buf.drain(..tag_len).collect();
                        out.push_str(&env_bytes);
                        *inside_envelope = false;
                    }
                    ActKind::OrphanOpen => {
                        tag_scan_buf.drain(..tag_len);
                        *suppressing_param_leak = true;
                        tracing::warn!(
                            "orphan tool-call leak in content stream; suppressing until close"
                        );
                    }
                    ActKind::OrphanClose => {
                        tag_scan_buf.drain(..tag_len);
                        // Stray close outside suppression — silently dropped.
                    }
                }
                continue;
            }
            None => {
                let buf_len = tag_scan_buf.len();
                let hold = tag_max.saturating_sub(1);
                if buf_len <= hold {
                    break;
                }
                let commit_to = buf_len - hold;
                let cut = tag_scan_buf
                    .char_indices()
                    .map(|(i, _)| i)
                    .take_while(|&i| i <= commit_to)
                    .last()
                    .unwrap_or(0);
                let emit: String = tag_scan_buf.drain(..cut).collect();
                out.push_str(&emit);
                break;
            }
        }
    }
    out
}

// Repetition-loop watchdog. Accumulates up to ~8 KB of recent
// post-detector content in `loop_scan_buf` and returns `true` when the
// most recent non-trivial line appears ≥ 4× in the tail — a signal the
// model is stuck in a degenerate prose loop. Conservative: ignores
// lines shorter than 15 trimmed chars, code-fence openers, and already
// triggered flags.
//
// MUST operate on post-detector Content only. Tool-call parameter
// values flow through the detector as structured chunks (not Content)
// — running this on raw delta would truncate a tool call whose arg
// legitimately repeats a short line (e.g. Rust source with
// `self.error.clear();` per method), leaving the client with an
// incomplete tool call.
//
// Window bumped 3 KB → 8 KB (2026-04-25, claude-export.txt
// failure): the export's "I'll create the project files and verify
// everything works:" phrase repeated 4× at end-of-stream, but earlier
// instances rolled out of the 3 KB window because each repetition was
// preceded by ~3 KB of source-dump prose. 8 KB keeps multi-paragraph
// repetitions in view across larger interstitials.
//
// Fuzzy comparison + substring scan (2026-04-25): the same export
// had its 4th repeat begin mid-line ("…everything works:        let
// body ="), defeating exact-line equality. We now compare lines
// using trimmed + lowercased + whitespace-collapsed equality, AND
// for ≥30-char candidate lines also count substring occurrences in
// the buffer (catches mid-line continuations of an otherwise
// repeating phrase).
//
// ── F7 (2026-04-26): cross-turn tool-arg-path stall guard ──
//
// Live evidence from `/workspace/atlas-opencode-dump-fix28.jsonl`
// showed the model writing the same `Cargo.toml` 7 times across 17
// turns when cargo wasn't installed and F6 (is_error capture) made
// it correctly recognise but futilely retry. F1-F5 catch per-
// response loops; F7 catches the per-conversation pattern by
// scanning message history before the request reaches the
// scheduler. At ≥3 same-bucket hits, append a system-reminder; at
// ≥5, escalate to a stop-tool-calls directive (the request still
// goes through, but the model is told plainly to respond in text).

// F14 (2026-04-26): raised from 3 → 4. AR2's survey: Gemini-CLI
// uses 5-consecutive, Anthropic's documented per-turn ceiling is
// ~10. Atlas at 3 was too aggressive — false-positives on
// legitimate "build / fix / build" cycles. 4 sits between the
// production references while still preventing the fix28
// 7-rewrite scenario.
pub const F7_STALL_WARN_THRESHOLD: u32 = 4;
pub const F7_STALL_REFUSE_THRESHOLD: u32 = 5;
const F7_BASH_COMMAND_PREFIX_LEN: usize = 80;
const F7_OTHER_ARG_FALLBACK_LEN: usize = 80;

/// Per-(tool_name, primary_arg) hit counts across the conversation.
pub type F7StallBuckets = std::collections::HashMap<(String, String), u32>;

/// F21 (2026-04-26): extract the FINAL meaningful command from a
/// Bash chain. Splits on `&&`, `||`, `;`, newlines. Drops `cd …`
/// boilerplate. Truncates the result to F7_BASH_COMMAND_PREFIX_LEN
/// characters (UTF-8-boundary safe). The returned string is the
/// F7 bucket key for Bash tool calls.
///
/// Examples:
///   "mkdir -p /tmp/x && cd /tmp/x && cargo init --name a"
///     → "cargo init --name a"
///   "mkdir -p /tmp/x/src && cd /tmp/x && cargo init --name a"
///     → "cargo init --name a"  (collapses with above)
///   "ls -la /tmp/x"
///     → "ls -la /tmp/x"
pub fn extract_bash_final_action(command: &str) -> String {
    // Split on shell-chain operators. We split on each character
    // class separately to keep it simple; any of the splitters
    // collapses adjacent empty pieces below.
    let parts: Vec<&str> = command
        .split(['&', '|', ';', '\n'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && !s.starts_with("cd ") && !s.starts_with("cd\t") && *s != "cd")
        .collect();
    let action = parts.last().copied().unwrap_or(command);
    let n = action.len().min(F7_BASH_COMMAND_PREFIX_LEN);
    let mut cut = n;
    while cut > 0 && !action.is_char_boundary(cut) {
        cut -= 1;
    }
    action[..cut].to_string()
}

/// Extract a "primary arg" string from a tool call's JSON arguments.
/// The choice of primary arg is per-tool: Write/Edit/Read use
/// `file_path`; Bash uses `command` (truncated so flag-only diffs
/// collapse); other tools fall back to the first non-empty
/// string-valued field in canonical key order.
/// F51 (2026-04-27): tool-name family classifier (SSOT). opencode
/// sends lowercase tool names (`write`, `bash`, `edit`); Claude
/// Code sends uppercase Anthropic-style. All match arms in F-fix
/// helpers must accept both. Centralised here so adding a new tool
/// family touches one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Bash,
    Write,
    Edit,
    Read,
    MultiEdit,
    Other,
}

pub fn classify_tool(name: &str) -> ToolKind {
    if name.eq_ignore_ascii_case("Bash") {
        ToolKind::Bash
    } else if name.eq_ignore_ascii_case("Write") {
        ToolKind::Write
    } else if name.eq_ignore_ascii_case("Edit") {
        ToolKind::Edit
    } else if name.eq_ignore_ascii_case("Read") {
        ToolKind::Read
    } else if name.eq_ignore_ascii_case("MultiEdit") {
        ToolKind::MultiEdit
    } else {
        ToolKind::Other
    }
}

pub fn primary_arg_for_tool(name: &str, args_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args_json).ok()?;
    let obj = v.as_object()?;
    let kind = classify_tool(name);
    let key_for_well_known = match kind {
        ToolKind::Write | ToolKind::Edit | ToolKind::Read | ToolKind::MultiEdit => {
            // opencode uses `filePath`; Claude Code uses `file_path`.
            // Try both — the helper at args_json -> obj is structured.
            if obj.get("file_path").and_then(|v| v.as_str()).is_some() {
                Some("file_path")
            } else if obj.get("filePath").and_then(|v| v.as_str()).is_some() {
                Some("filePath")
            } else {
                Some("file_path")
            }
        }
        ToolKind::Bash => Some("command"),
        ToolKind::Other => None,
    };
    if let Some(k) = key_for_well_known {
        let val = obj.get(k).and_then(|v| v.as_str())?;
        if matches!(kind, ToolKind::Bash) {
            // F21 (2026-04-26): bucket on the FINAL command in the
            // shell chain rather than the first 80 chars. Splits on
            // `&&`, `||`, `;`, `\n`. Drops leading `cd …` segments
            // (which the model uses as boilerplate). All five fix30
            // cargo-init variants (different mkdir prefixes) collapse
            // to the same bucket "cargo init --name axum_echo_server".
            return Some(extract_bash_final_action(val));
        }
        return Some(val.to_string());
    }
    // Fallback: first non-empty string-valued field, sorted by key.
    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    for k in keys {
        if let Some(s) = obj.get(k).and_then(|v| v.as_str())
            && !s.is_empty()
        {
            let n = s.len().min(F7_OTHER_ARG_FALLBACK_LEN);
            let mut cut = n;
            while cut > 0 && !s.is_char_boundary(cut) {
                cut -= 1;
            }
            return Some(format!("{}={}", k, &s[..cut]));
        }
    }
    None
}
