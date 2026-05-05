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
}

/// Outputs of [`build_msg_entries`]. Bundled as a struct because
/// the caller threads each field through five later phases.
pub(super) struct BuildOut {
    pub(super) messages: Vec<MsgEntry>,
    pub(super) cwd_hint: Option<String>,
    pub(super) image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub(super) image_pad_counts: Vec<usize>,
    pub(super) consecutive_tool_errors: u32,
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

    // Find the last real user query (not a tool response). Only
    // assistant messages AFTER this index get the empty `<think>`
    // wrapper (Jinja template pattern).
    let last_query_index = req
        .messages
        .iter()
        .rposition(|m| m.role == "user" && !m.content.text.starts_with("<tool_response>"))
        .unwrap_or(req.messages.len().saturating_sub(1));

    for (msg_idx, m) in req.messages.iter().enumerate() {
        let mut text = m.content.text.clone();

        // Historical assistant messages after the last user query
        // get an empty think block to match training format.
        // Skip when thinking is suppressed in tool turns (the
        // Jinja template handles think wrapping itself).
        let thinking_suppressed = tools_active && !state.behavior.thinking_in_tools;
        if m.role == "assistant"
            && state.tokenizer.supports_thinking()
            && msg_idx > last_query_index
            && !thinking_suppressed
        {
            text = format!("<think>\n\n</think>\n\n{text}");
        }

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

        // Tool-response messages: pass raw content; Jinja template
        // handles `<tool_response>` wrapping and consecutive
        // grouping.
        if tools_active && m.role == "tool" {
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
            });
            continue;
        }

        let image_count = m.content.images.len();
        messages.push(MsgEntry {
            role: m.role.clone(),
            content: text,
            tool_calls: tool_calls_json,
            image_count,
        });
        if !m.content.images.is_empty() {
            for img_uri in &m.content.images {
                all_images.push(img_uri.clone());
                image_pad_counts.push(0);
            }
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

    // Preprocess images if a vision config is available.
    let mut image_pixels: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    if !all_images.is_empty()
        && let Some(vcfg) = &state.vision_config
    {
        for (idx, uri) in all_images.iter().enumerate() {
            match spark_model::vision_preprocess::preprocess_image(uri, vcfg) {
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

    Ok(BuildOut {
        messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
        consecutive_tool_errors,
    })
}
