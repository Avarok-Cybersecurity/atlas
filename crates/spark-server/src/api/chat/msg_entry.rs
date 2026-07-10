// SPDX-License-Identifier: AGPL-3.0-only
//
// `MsgEntry` + the pre-loop builder that turns the inbound
// `ChatCompletionRequest.messages` into the local representation
// used by every downstream phase (json_messages, loop detector,
// task pin, observation mask, …).
//
// Lifted out of `chat::chat_completions_inner` (wave 4g) so the
// orchestrator stays under the 500-LoC cap.

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;
use crate::openai::ChatCompletionRequest;

use super::super::compact::openai_error_response;

/// Per-message data: role, content text, optional structured
/// `tool_calls`, and image-part count for the Jinja vision-marker
/// expansion. `pub(super)` so `chat::chat_completions_inner` and
/// the other `chat/*` sub-files can read every field.
pub(super) struct MsgEntry {
    pub(super) role: String,
    pub(super) content: String,
    /// Structured tool_calls for the Jinja template (arguments
    /// pre-parsed to dicts).
    pub(super) tool_calls: Option<Vec<serde_json::Value>>,
    /// Number of image content parts on this message. When > 0
    /// the json_messages builder emits a structured content array
    /// so the Jinja template can render
    /// `<|vision_start|><|image_pad|><|vision_end|>` markers.
    pub(super) image_count: usize,
    /// Historical reasoning trace from a prior assistant turn (the
    /// `<think>...</think>` body). Forwarded from `IncomingMessage`
    /// and passed to the Jinja template so the template can
    /// rehydrate the historical `<think>` block. Empty/None ⇒ no
    /// `<think>` block emitted for this message — prevents the
    /// empty-`<think></think>` poisoning pattern that triggers
    /// premature `<|im_end|>` (vLLM/SGLang #131, MLC commit d75d64e).
    pub(super) reasoning_content: Option<String>,
}

/// Outputs of [`build_msg_entries`]. Bundled as a struct because
/// the caller threads each field through five later phases.
pub(super) struct BuildOut {
    pub(super) messages: Vec<MsgEntry>,
    pub(super) cwd_hint: Option<String>,
    pub(super) image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub(super) image_pad_counts: Vec<usize>,
}

#[allow(clippy::result_large_err)]
pub(super) fn build_msg_entries(
    state: &Arc<AppState>,
    req: &ChatCompletionRequest,
    tools_active: bool,
) -> Result<BuildOut, Response> {
    let mut messages: Vec<MsgEntry> = Vec::with_capacity(req.messages.len());
    let mut all_images: Vec<String> = Vec::new();
    let mut image_pad_counts: Vec<usize> = Vec::new();
    let mut consecutive_tool_errors: u32 = 0;
    // BW1 bash-wandering watchdog: tally tool-call productivity across the
    // conversation so a steering nudge can fire if the agent explores/runs
    // many commands without ever writing the deliverable (gap #9).
    let mut total_tool_calls: usize = 0;
    let mut productive_tool_calls: usize = 0;
    // P1-6 (2026-07-09): (index into `messages`, pre-hint original
    // text) for every tool-result entry, for duplicate-error masking
    // after the loop. Comparison must see the ORIGINAL text — the
    // injected hints vary with the escalation counter and would break
    // exact-match grouping.
    let mut tool_result_originals: Vec<(usize, String)> = Vec::new();

    // F6 (2026-05-26): `last_query_index` was previously used to gate
    // an empty `<think>\n\n</think>\n\n` injection for historical
    // assistant turns. The Jinja template already does this gating
    // itself (via its own `ns.last_query_index` computation) and the
    // injection here was the source of empty-think poisoning. Removed.
    for (msg_idx, m) in req.messages.iter().enumerate() {
        let _ = msg_idx;
        let text = m.content.text.clone();

        // Preserve structured tool_calls for the Jinja template.
        // Always extract from assistant messages — past turns may
        // carry tool_calls that the template MUST render even when
        // the current request didn't pass `tools`.
        let tool_calls_json = if m.role == "assistant" {
            m.tool_calls.as_ref().and_then(|tcs| {
                if tcs.is_empty() {
                    return None;
                }
                let parsed: Vec<serde_json::Value> = tcs
                    .iter()
                    .map(|tc| {
                        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::Value::Object(Default::default()));
                        serde_json::json!({
                            "id": tc.id.as_deref().unwrap_or(""),
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": args
                            }
                        })
                    })
                    .collect();
                Some(parsed)
            })
        } else {
            None
        };

        // BW1: tally tool-call productivity (write/edit/build-run vs explore).
        if m.role == "assistant"
            && let Some(ref tcs) = m.tool_calls
        {
            for tc in tcs {
                total_tool_calls += 1;
                let args: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                if crate::hint_injector::tool_call_is_productive(&tc.function.name, &args) {
                    productive_tool_calls += 1;
                }
            }
        }

        // Tool-response messages: pass raw content; Jinja template
        // handles `<tool_response>` wrapping and consecutive
        // grouping.
        if tools_active && m.role == "tool" {
            let mut text = text;
            // P1-6 (2026-07-09): record the pre-hint original at the
            // index this entry is about to occupy.
            tool_result_originals.push((messages.len(), text.clone()));
            if crate::hint_injector::looks_like_error(&text) {
                consecutive_tool_errors += 1;
                crate::hint_injector::inject_hints(&mut text, consecutive_tool_errors);
            } else {
                consecutive_tool_errors = 0;
            }
            messages.push(MsgEntry {
                role: "tool".into(),
                content: text,
                tool_calls: None,
                image_count: 0,
                reasoning_content: None,
            });
            continue;
        }

        let image_count = m.content.images.len();
        // Wave 3 (2026-05-26): `ATLAS_STRIP_REASONING_HISTORY=1` drops
        // historical reasoning_content entirely. Matches MLC commit
        // d75d64e (Apr 2026) `strip_reasoning_in_history` for qwen3,
        // whose PR description matches Atlas's Wave-1 failure mode
        // verbatim: echoing prior `<think>` traces makes the next turn
        // emit `<|im_end|>` prematurely AND seeds loop-attractor drift
        // on prior-failed-attempt token patterns (the `lean://` loop
        // observed in the Wave-1 opencode probe).
        let strip_reasoning = std::env::var("ATLAS_STRIP_REASONING_HISTORY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        messages.push(MsgEntry {
            role: m.role.clone(),
            content: text,
            tool_calls: tool_calls_json,
            image_count,
            // F1: forward reasoning_content for assistant messages only.
            // Wave 3: when strip_reasoning=true, drop it for ALL turns,
            // forcing the template back to the pre-F1 "clean content
            // only" rendering shape — but without re-introducing the
            // empty-`<think>\n\n</think>\n\n` poisoning, because F6's
            // template change skips the wrapper when reasoning_content
            // is empty.
            reasoning_content: if m.role == "assistant" && !strip_reasoning {
                m.reasoning_content
                    .as_ref()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            } else {
                None
            },
        });
        if !m.content.images.is_empty() {
            for img_uri in &m.content.images {
                all_images.push(img_uri.clone());
                image_pad_counts.push(0);
            }
        }
    }

    // P1-6 (2026-07-09): duplicate-error observation masking
    // (arXiv:2508.21433 pattern). Feeding an identical error text back
    // verbatim N times reinforces the failing-call attractor (45k
    // collapse: 6x "BadResource: FileSystem.readFile (/home/nologik)").
    // Mask the OLDER occurrences of a repeated error-shaped tool
    // result, keeping only the NEWEST verbatim (hints attach to the
    // newest, which is preserved untouched). Kill-switch
    // ATLAS_NO_ERROR_DEDUP=1 restores verbatim history. MUST run
    // before the vacuous-system removal below — the recorded indices
    // refer to the un-shifted `messages` vec.
    if tools_active && !error_dedup_disabled() {
        for (idx, replacement) in duplicate_error_masks(&tool_result_originals) {
            messages[idx].content = replacement;
        }
    }

    // Extract working directory from the system message if present.
    let cwd_hint: Option<String> = messages.iter().find(|m| m.role == "system").and_then(|m| {
        for line in m.content.lines() {
            let lower = line.to_lowercase();
            if (lower.contains("working directory")
                || lower.contains("working_directory")
                || lower.contains("cwd:"))
                && let Some(pos) = line.find(':')
            {
                let path = line[pos + 1..]
                    .trim()
                    .trim_matches(|c| c == '`' || c == '"' || c == '\'');
                if !path.is_empty() {
                    return Some(path.to_string());
                }
            }
        }
        None
    });

    // Inject CWD hint into the system message (NOT tool definitions —
    // those go to the Jinja template).
    if tools_active && let Some(ref cwd) = cwd_hint {
        let hints = format!("\n<environment>\nworking_directory: {cwd}\n</environment>");
        if let Some(first) = messages.first_mut()
            && first.role == "system"
        {
            first.content.push_str(&hints);
        }
    }

    // Neutralize a content-free leading system message. Clients (notably
    // Open WebUI's empty RAG/context template) inject a system message
    // carrying NO instruction — e.g. `"User Context:\n\n"` (trims to the
    // bare label `User Context:`). Models react to a content-free system
    // directive by producing terse / prematurely-terminated output
    // (isolated 2026-05-17: removing it 3x'd generation length on the
    // 3D-chess prompt). We can't fix the client, so Atlas adapts: treat
    // such a message as absent so a degenerate client prompt can't poison
    // generation. Conservative — only an empty body or a single short
    // bare `Label:` line qualifies; any substantive prompt is untouched.
    if messages
        .first()
        .is_some_and(|m| m.role == "system" && is_vacuous_system_content(&m.content))
    {
        let removed = messages.remove(0);
        tracing::info!(
            dropped = %removed.content.trim(),
            "Dropped content-free client system message (would bias the model toward terse output)"
        );
    }

    // Preprocess images if a vision config is available.
    let mut image_pixels: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    if !all_images.is_empty()
        && let Some(vcfg) = &state.vision_config
    {
        for (idx, uri) in all_images.iter().enumerate() {
            match spark_model::vision_preprocess::preprocess_image_with_max_pixels(
                uri,
                vcfg,
                state.vision_max_pixels,
            ) {
                Ok((pixels, grid_h, grid_w)) => {
                    image_pad_counts[idx] = spark_model::vision_preprocess::image_pad_count(
                        grid_h,
                        grid_w,
                        vcfg.spatial_merge_size,
                    );
                    image_pixels.push((pixels, grid_h, grid_w));
                }
                Err(e) => {
                    return Err(openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!("Image decode error: {e}"),
                    ));
                }
            }
        }
    }
    // If no vision_config (text-only model), image_pad_counts stays
    // 0 and images are silently dropped on the encoder side.

    // BW1 bash-wandering watchdog: if the agent has run many tool calls with
    // no productive file output, append a steering nudge to the most recent
    // tool response (what the model reads just before its next action). Gated
    // by ATLAS_BASH_WANDER_WATCHDOG (PCND, default-off).
    if tools_active
        && let Some(hint) =
            crate::hint_injector::bash_wander_hint(total_tool_calls, productive_tool_calls)
        && let Some(last_tool) = messages.iter_mut().rev().find(|e| e.role == "tool")
    {
        last_tool.content.push_str(&hint);
    }

    Ok(BuildOut {
        messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
    })
}

/// True when a system message carries no actual instruction and should
/// be treated as absent. Conservative by design — a substantive prompt
/// must never be stripped:
///   * empty / whitespace-only body, OR
///   * a single short bare label line ending in ':' with nothing after
///     it (e.g. `User Context:`, `Context:`, `System:`) — the residue
///     of an empty client template (Open WebUI's RAG/context block).
/// Anything multi-line, or with any text past the colon, is a real
/// prompt and returns false.
fn is_vacuous_system_content(content: &str) -> bool {
    let t = content.trim();
    if t.is_empty() {
        return true;
    }
    if !t.contains('\n') && t.len() <= 32 && t.ends_with(':') {
        let label = &t[..t.len() - 1];
        return !label.is_empty()
            && label
                .chars()
                .all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '_' || c == '-');
    }
    false
}

/// P1-6 (2026-07-09): kill-switch — `ATLAS_NO_ERROR_DEDUP=1` restores
/// verbatim duplicate-error history (disables the masking pass).
fn error_dedup_disabled() -> bool {
    std::env::var("ATLAS_NO_ERROR_DEDUP").as_deref() == Ok("1")
}

/// P1-6 (2026-07-09): duplicate-error observation masking.
///
/// Input: `(message_index, original_pre_hint_text)` for every
/// tool-result entry, in conversation order. Output: `(message_index,
/// replacement_text)` for the OLDER members of each duplicate-error
/// group; the newest member of each group stays verbatim. Two
/// tool results are duplicates when BOTH are error-shaped
/// (`crate::hint_injector::looks_like_error`) AND either equal after
/// trim or Jaccard >= 0.9 over 4-gram shingles (the loop_detector
/// measure — SSOT). Successful outputs never participate: identical
/// success observations (e.g. repeated `ls`) are legitimate.
fn duplicate_error_masks(tool_results: &[(usize, String)]) -> Vec<(usize, String)> {
    const NEAR_DUP_JACCARD: f64 = 0.9;
    let errors: Vec<(usize, &str)> = tool_results
        .iter()
        .filter(|(_, t)| crate::hint_injector::looks_like_error(t))
        .map(|(i, t)| (*i, t.trim()))
        .collect();
    if errors.len() < 2 {
        return Vec::new();
    }
    let shingle_sets: Vec<_> = errors
        .iter()
        .map(|(_, t)| crate::loop_detector::shingle_set(t))
        .collect();
    // First-match grouping against each group's first member. Short
    // errors (< 4 tokens) have empty shingle sets — jaccard() returns
    // 0.0 for those, so they group via exact-trim match only.
    let mut groups: Vec<Vec<usize>> = Vec::new(); // indices into `errors`
    for i in 0..errors.len() {
        let group = groups.iter_mut().find(|g| {
            let rep = g[0];
            errors[rep].1 == errors[i].1
                || crate::loop_detector::jaccard(&shingle_sets[rep], &shingle_sets[i])
                    >= NEAR_DUP_JACCARD
        });
        match group {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    let mut masks = Vec::new();
    for g in &groups {
        let n = g.len();
        if n < 2 {
            continue;
        }
        // Keep the LAST (newest) verbatim; mask the earlier ones.
        for (k, &ei) in g.iter().take(n - 1).enumerate() {
            masks.push((
                errors[ei].0,
                format!("[same error as below, attempt {} of {}]", k + 1, n),
            ));
        }
    }
    masks
}

#[cfg(test)]
#[path = "msg_entry_tests.rs"]
mod msg_entry_tests;
