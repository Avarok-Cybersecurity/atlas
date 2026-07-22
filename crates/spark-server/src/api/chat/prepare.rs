// SPDX-License-Identifier: AGPL-3.0-only
//
// Shared prompt-preparation front half of the chat pipeline: tool
// gating + tool-prompt injection + MsgEntry build (image preprocess,
// cwd extraction) + thinking resolution + Jinja render. Extracted from
// `chat_completions_inner` so `/v1/messages/count_tokens` counts
// against the EXACT prompt the serving path renders instead of a
// divergent third lowering.

use axum::response::Response;
use std::sync::Arc;

use crate::AppState;
use crate::ir::ChatRequest;

use super::{msg_entry, template, thinking};

/// Outputs of [`prepare_chat_prompt`].
pub(crate) struct PreparedChat {
    pub(crate) tools_active: bool,
    pub(crate) cwd_hint: Option<String>,
    pub(crate) image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub(crate) prompt_tokens: Vec<u32>,
    pub(crate) enable_thinking: bool,
    pub(crate) thinking_budget: Option<u32>,
}

/// Run the prompt-affecting phases against the IR envelope. Mutates
/// `req.messages` (tool-prompt injection). Everything here must stay
/// deterministic for a given `(req, state)` — the rendered
/// `prompt_tokens` are the kv-cache prefix.
#[allow(clippy::result_large_err)]
pub(crate) fn prepare_chat_prompt(
    state: &Arc<AppState>,
    req: &mut ChatRequest,
) -> Result<PreparedChat, Response> {
    // Tool-active gating.
    let tools_active = state.tool_call_parser.is_some()
        && !req.tools.is_empty()
        && !req.tool_choice.as_ref().is_some_and(|tc| tc.is_none());

    // ST-995 fix: restore the parser-specific behavioral system prompt #90
    // removed. For the hermes parser this is the canonical NousResearch
    // function-calling prompt ("you MAY call one or more functions...
    // don't make assumptions"), which the GDN model needs to correctly
    // DECLINE on irrelevance prompts. With it (and compact tool-JSON)
    // hallucination returns to ~96 (vs 30/64 without).
    if tools_active && let Some(ref parser) = state.tool_call_parser {
        let default_choice = crate::tool_parser::ToolChoice::Mode("auto".to_string());
        let tool_choice = req.tool_choice.as_ref().unwrap_or(&default_choice);
        let tool_prompt = parser.system_prompt(&req.tools, tool_choice);
        if let Some(first) = req
            .messages
            .first_mut()
            .filter(|m| m.role == crate::ir::Role::System)
        {
            first.prepend_text(&format!("{tool_prompt}\n\n"));
        } else {
            req.messages
                .insert(0, crate::ir::Message::synthetic_system(tool_prompt));
        }
    }

    tracing::info!(
        "Request: model={}, messages={}, tools={}, tools_active={}, tool_choice={:?}, stream={}, temp={:?}, max_tokens={}, freq_pen={:?}, rep_pen={:?}",
        req.model,
        req.messages.len(),
        req.tools.len(),
        tools_active,
        req.tool_choice,
        req.stream,
        req.sampling.temperature,
        req.max_tokens,
        req.sampling.frequency_penalty,
        req.sampling.repetition_penalty,
    );

    // ── Phase 1: build MsgEntry vec + image preprocess + cwd ────
    let msg_entry::BuildOut {
        messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
    } = msg_entry::build_msg_entries(
        state.vision_config.as_ref(),
        state.vision_max_pixels,
        &req.messages,
        tools_active,
    )?;

    // ── Phase 1.5 + 2: thinking directive + resolution (pre-template) ─
    // The client's per-request directive (resolved at the API edge) wins;
    // when the client is silent the server-level
    // --default-chat-template-kwargs directive applies; when that too is
    // unspecified, MODEL.toml decides inside resolve_thinking.
    let mut thinking_directive = req.thinking;
    if !thinking_directive.is_explicit() {
        thinking_directive = state.default_thinking;
    }
    let (enable_thinking, thinking_budget) = thinking::resolve_thinking(
        state,
        thinking_directive,
        req.max_tokens as u32,
        tools_active,
    );

    // ── Phase 5: render Jinja template + image-pad expansion ────
    let template::TemplateOut {
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    } = template::render_template(
        state,
        &req.tools,
        &messages,
        &image_pad_counts,
        enable_thinking,
        thinking_budget,
        tools_active,
    )?;

    Ok(PreparedChat {
        tools_active,
        cwd_hint,
        image_pixels,
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    })
}
