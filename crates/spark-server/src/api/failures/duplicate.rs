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
    F23ProgressMetrics, F29EnvironmentFact, F37FailureClass, F39FailureCache,
    F39PermanentFailureMatch, append_f7_reminder_to_last_user, build_f7_stall_reminder,
    collect_f7_stall_buckets, f23_build_reminder, f23_normalize_and_hash, f23_refuse_threshold,
    f23_score_progress, f23_warn_threshold, f28_text_looks_like_error,
    f29_extract_binary_from_error_line, f29_extract_environment_facts,
    f29_inject_environment_facts, f31_inject_hard_refusal, f32_reposition_failed_tool_result,
    f37_classify_failure, f39_build_circuit_breaker_banner, f39_build_failure_cache,
    f39_class_label, f39_detect_recent_retries, f39_extract_binary_name,
    f44_check_permanent_failure, f60_disable_mtp_for_request, prepend_reminder_to_system,
    recent_message_is_tool_error,
};

// Re-export sibling helpers via crate::api::* for short paths.
use super::super::inference_types::*;

pub struct F49DuplicateWrite {
    pub file_path: String,
    pub prior_count: u32,
}

pub fn f49_extract_write_path_and_content(name: &str, args_json: &str) -> Option<(String, u64)> {
    // F51: case-insensitive tool family check. opencode uses
    // lowercase `write`/`edit`; Claude Code uses uppercase.
    let kind = classify_tool(name);
    if !matches!(kind, ToolKind::Write | ToolKind::Edit | ToolKind::MultiEdit) {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(args_json).ok()?;
    let obj = v.as_object()?;
    let path = obj
        .get("file_path")
        .or_else(|| obj.get("filePath"))
        .and_then(|v| v.as_str())?;
    // For Write: hash the `content` parameter. For Edit/MultiEdit:
    // hash the (oldString, newString) pair (the substitution is the
    // unique identity of the edit; rewriting the same Edit twice
    // means the model thinks the prior edit didn't apply). opencode
    // also uses `oldString`/`newString` keys.
    if matches!(kind, ToolKind::Write) {
        let content = obj.get("content").and_then(|v| v.as_str()).unwrap_or("");
        return Some((path.to_string(), f23_normalize_and_hash(content)));
    }
    obj.get("oldString")
        .and_then(|o| o.as_str())
        .zip(obj.get("newString").and_then(|n| n.as_str()))
        .map(|(o, n)| {
            (
                path.to_string(),
                f23_normalize_and_hash(&format!("{o}\u{1F}{n}")),
            )
        })
}

pub fn f49_detect_duplicate_writes(
    messages: &[crate::openai::IncomingMessage],
) -> Vec<F49DuplicateWrite> {
    // Build the seen-set from ALL prior assistant turns EXCEPT the
    // most recent. Then check the most recent's tool_calls against
    // that set.
    let mut last_asst_idx: Option<usize> = None;
    for (i, m) in messages.iter().enumerate().rev() {
        if m.role == "assistant" && m.tool_calls.as_ref().is_some_and(|t| !t.is_empty()) {
            last_asst_idx = Some(i);
            break;
        }
    }
    let Some(last_idx) = last_asst_idx else {
        return Vec::new();
    };
    let mut seen: std::collections::HashMap<(String, u64), u32> = std::collections::HashMap::new();
    for (i, m) in messages.iter().enumerate() {
        if i >= last_idx {
            break;
        }
        if m.role != "assistant" {
            continue;
        }
        let Some(tcs) = &m.tool_calls else { continue };
        for tc in tcs {
            let name = tc.function.name.clone();
            if let Some((path, hash)) =
                f49_extract_write_path_and_content(&name, &tc.function.arguments)
            {
                *seen.entry((path, hash)).or_insert(0) += 1;
            }
        }
    }
    let last = &messages[last_idx];
    let Some(tcs) = &last.tool_calls else {
        return Vec::new();
    };
    let mut hits: Vec<F49DuplicateWrite> = Vec::new();
    let mut emitted: std::collections::HashSet<(String, u64)> = std::collections::HashSet::new();
    for tc in tcs {
        let name = tc.function.name.clone();
        let Some((path, hash)) = f49_extract_write_path_and_content(&name, &tc.function.arguments)
        else {
            continue;
        };
        let key = (path.clone(), hash);
        if !emitted.insert(key.clone()) {
            continue;
        }
        if let Some(prior_count) = seen.get(&key) {
            hits.push(F49DuplicateWrite {
                file_path: path,
                prior_count: *prior_count,
            });
        }
    }
    hits
}

// F50 (2026-04-27): when F49 trips AND there's an EARLIER [tool error]
// in the conversation that's no longer at the tail, append a copy of
// that original error at the conversation tail with an
// `[atlas-original-error]` prefix. Counters Lost-in-the-Middle on
// the specific opencode pattern: the original `couldn't read
// src/main.rs` error scrolled past attention focus when 4 successful
// Write tool_results pushed it backward, so the model fell back to
// generic syntax paraphrasing.
pub fn f50_append_original_error(messages: &mut Vec<crate::openai::IncomingMessage>) -> bool {
    // Find the FIRST role:tool message whose content has an error
    // signature. Skip the most recent few tool_results (assume those
    // are the misdiagnosed-rewrite results). We anchor on the
    // OLDEST error so the model is reminded of the root cause.
    let mut original_idx: Option<usize> = None;
    for (i, m) in messages.iter().enumerate() {
        if m.role == "tool" && f28_text_looks_like_error(&m.content.text) {
            original_idx = Some(i);
            break;
        }
    }
    let Some(orig_idx) = original_idx else {
        return false;
    };
    // Skip if the original error is already at the tail (or near it).
    if messages.len().saturating_sub(orig_idx + 1) < 2 {
        return false;
    }
    // Don't double-emit if we've already injected a tail reminder
    // for this conversation (idempotent).
    if let Some(last) = messages.last()
        && last.role == "tool"
        && last.content.text.starts_with("[atlas-original-error]")
    {
        return false;
    }
    // Anchor to the most recent assistant turn's first tool_call id
    // so transport-level frame validators don't drop the synthesised
    // tool_result. We synthesise as a `role: tool` reply.
    let mut anchor_id: Option<String> = None;
    for m in messages.iter().rev() {
        if m.role == "assistant" {
            if let Some(tcs) = &m.tool_calls
                && let Some(first) = tcs.first()
                && let Some(id) = first.id.as_ref()
            {
                anchor_id = Some(id.clone());
            }
            break;
        }
    }
    let Some(tool_call_id) = anchor_id else {
        return false;
    };
    let original_text = messages[orig_idx].content.text.clone();
    let body = format!(
        "[atlas-original-error] You have lost focus on the ORIGINAL error that prompted this work. Re-read it carefully — it likely cites a specific file, line, or symbol that needs to change:\n\n\
         ----- BEGIN ORIGINAL ERROR -----\n\
         {}\n\
         ----- END ORIGINAL ERROR -----\n\n\
         Address THAT error specifically (the file/line it cites), not a generic syntax issue. If you cannot resolve it, reply to the user in plain text describing what is blocking.",
        original_text.trim()
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
    messages.push(synth);
    true
}

pub fn f49_build_banner(hits: &[F49DuplicateWrite]) -> String {
    let lines: Vec<String> = hits
        .iter()
        .map(|h| {
            format!(
                "- {} (already written {} time{} with byte-identical content)",
                h.file_path,
                h.prior_count,
                if h.prior_count == 1 { "" } else { "s" }
            )
        })
        .collect();
    format!(
        "<atlas_duplicate_write>\n\
         CRITICAL: You are about to write the SAME content to the following file(s) that you ALREADY wrote earlier in this conversation. The new write is byte-identical to a prior write:\n\
         {}\n\n\
         The previous write succeeded — re-running it CANNOT fix anything. This means your diagnosis of the underlying error is WRONG. Re-read the ORIGINAL error message that prompted this rewrite (look earlier in the conversation for `[tool error]`, `error:`, or a stderr-like block). Pay attention to the SPECIFIC FILENAME, LINE NUMBER, or symbol it cites — that is what needs to change, not this file.\n\n\
         Do NOT write this file again with the same content. Either (a) write DIFFERENT content addressing the actual error cited, OR (b) reply to the user in plain text describing what you cannot resolve.\n\
         </atlas_duplicate_write>",
        lines.join("\n")
    )
}

/// Strip redundant bare-XML tool-call leaks from assistant content
/// before the chat template renders prior-turn history.
///
/// Context (investigation under `/workspace/.claude/plans/
/// so-i-also-saw-validated-catmull.md`, 2026-04-24):
/// an opencode agentic session against Qwen3.5-35B-A3B-FP8 degenerated
/// at turn 3 — bare `<write>` opener + whitespace → voluntary EOS.
/// The Phase-1 experiment (replay seq=3 verbatim on a wiped-cache
/// server) reproduced the collapse, proving the failure is
/// prompt-induced, not state-induced. The toxic input:
///
/// ```text
/// message[4] = assistant:
///   content: "…Let me create the Rust calculator module with the
///             source files first.\n\n<read><filePath>…</filePath>
///             <offset>1</offset><limit>100</limit></read>"
///   tool_calls: [{name: "write", …}]
/// ```
///
/// The XML `<read>` block in `content` is a hallucinated mirror of
/// (and in fact contradicts) the real `tool_calls[0]` field. The
/// jinja template renders `content` before `<tool_call>` blocks, so
/// by turn 3 the model sees BOTH formats presented as valid prior
/// behaviour and collapses when it tries to reproduce the XML form.
///
/// This helper removes any `<NAME>…</NAME>` block from `content`
/// where:
/// - `NAME` case-insensitively matches a declared tool's function
///   name (≥3 chars to avoid prose false positives),
/// - a matching close tag `</NAME>` exists,
/// only runs on assistant messages that ALSO carry a real
/// `tool_calls` field (else the block might be the only actionable
/// artefact and shouldn't be removed — see fix18's salvage path).
///
/// Prose outside these blocks is preserved verbatim.
pub fn strip_xml_leaks_from_assistant_content(
    content: &str,
    tool_defs: &[tool_parser::ToolDefinition],
) -> String {
    if content.is_empty() {
        return String::new();
    }
    // Build the leak-name set: every declared tool's name PLUS the
    // known agent-harness syntactic-sugar tags that opencode/Claude
    // Code clients emit and that the model mimics in prose. Adding
    // these prevents the prose-form `<task><file>…</file><content>…
    // </content></task>` envelope (dump 2026-04-25 seq=104..111)
    // from polluting the next turn's prompt, which previously
    // taught the model that emitting the prose envelope is a valid
    // tool-call substitute (Phase-1 leak-then-collapse pattern,
    // see history/SSM_CATASTROPHIC_FORGETTING_TODO.md).
    const HARNESS_TAGS: &[&str] = &["task", "file", "content", "description", "prompt", "glob"];
    let mut leak_names: Vec<String> = tool_defs
        .iter()
        .filter_map(|t| {
            let n = &t.function.name;
            if n.len() >= 3 {
                Some(n.to_ascii_lowercase())
            } else {
                None
            }
        })
        .collect();
    for tag in HARNESS_TAGS {
        if !leak_names.iter().any(|n| n == tag) {
            leak_names.push((*tag).to_string());
        }
    }

    let mut out = String::with_capacity(content.len());
    let mut cursor = 0usize;
    let lower = content.to_ascii_lowercase();
    while cursor < content.len() {
        // Find the earliest leak-block start at or after cursor.
        let mut best: Option<(usize, usize, usize)> = None; // (start, open_len, close_end)
        for name_lower in &leak_names {
            let open = format!("<{name_lower}>");
            let close = format!("</{name_lower}>");
            let Some(rel_start) = lower[cursor..].find(&open) else {
                continue;
            };
            let start = cursor + rel_start;
            let body_start = start + open.len();
            let Some(rel_close) = lower[body_start..].find(&close) else {
                continue;
            };
            let close_end = body_start + rel_close + close.len();
            match best {
                Some((bs, _, _)) if bs <= start => {}
                _ => best = Some((start, open.len(), close_end)),
            }
        }
        match best {
            Some((start, _open_len, close_end)) => {
                // Emit prose before the leak, then skip the block.
                out.push_str(&content[cursor..start]);
                cursor = close_end;
            }
            None => {
                out.push_str(&content[cursor..]);
                break;
            }
        }
    }
    // Collapse the typical "\n\n…leak removed…\n\n" into a single
    // blank line so the rendered prompt doesn't carry a suspicious
    // run of empties where the block used to be.
    let trimmed = out.trim_end_matches([' ', '\t']);

    trimmed.replace("\n\n\n", "\n\n")
}

/// Flush anything held in the sanitizer's tail buffer at stream end.
/// Drops content if suppression is still active (no close arrived) or if
/// the remaining bytes look like an incomplete tag opener.
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
    // Ambiguity threshold: longest possible tag this parser watches for.
    // A bare `<` tail shorter than that could still be the start of one
    // of our markers, so we drop it to avoid mid-tag flush. If the
    // parser declares no markers, the tail is by definition not tag-
    // related — emit verbatim.
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
