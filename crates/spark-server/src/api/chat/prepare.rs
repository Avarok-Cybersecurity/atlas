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

// Opt-in per-request phase timing for the chat request path
// (`ATLAS_CHAT_PHASE_TIMING=1`, strict `== "1"`).
//
// Exists to localize a measured ~205 ms that `/v1/chat/completions` costs over
// `/v1/completions` for identical content on a SHORT, TOOL-LESS prompt. It is
// per-request CPU rather than prefill (it survives at p50 across warm reps),
// and it is not template compilation (the minijinja env is precompiled) nor the
// auto-compact double render (disabled by default). At ~350-450 ms/turn in the
// harness's shape that is ~9-12% of the wall, so it is worth an instrument.
//
// Off by default, so the production path is unchanged. The flag itself is
// `ChatLevers::phase_timing`, resolved when `AppState` is built — a
// `OnceLock` here would have made the instrument un-toggleable for the life
// of the process, including across a model swap.

/// Outputs of [`prepare_chat_prompt`].
pub(crate) struct PreparedChat {
    pub(crate) tools_active: bool,
    pub(crate) cwd_hint: Option<String>,
    pub(crate) image_pixels: Vec<spark_model::VisionItem>,
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

    // Preserve parser-owned prompts (including Hermes and TSCG). Native
    // Qwen checkpoint templates already render their tool instructions;
    // adding the parser block there duplicates the schema and changes the
    // request with unrelated behavioral guidance.
    if tools_active && let Some(ref parser) = state.tool_call_parser {
        let default_choice = crate::tool_parser::ToolChoice::Mode("auto".to_string());
        let tool_choice = req.tool_choice.as_ref().unwrap_or(&default_choice);
        let tool_prompt = parser_tool_prompt(
            parser.as_ref(),
            &req.tools,
            tool_choice,
            &state.chat.prompt,
            state.tokenizer.uses_native_qwen_tool_template(),
        );
        inject_tool_system_prompt(&mut req.messages, tool_prompt);
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

    let _t_phase = std::time::Instant::now();

    // ── Phase 1: build MsgEntry vec + image preprocess + cwd ────
    let msg_entry::BuildOut {
        messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
    } = msg_entry::build_msg_entries(
        state.vision_config.as_ref(),
        state.vision_max_pixels,
        &state.remote_image_policy,
        &msg_entry::VideoDecode {
            ffmpeg: &state.video_ffmpeg,
            fps: state.video_fps,
        },
        &req.messages,
        tools_active,
        &state.chat,
        state.tokenizer.uses_deepseek_v4_encoding(),
    )?;
    let us_msg_entry = _t_phase.elapsed().as_micros();

    // ── Phase 1.5 + 2: thinking directive + resolution (pre-template) ─
    // The client's per-request directive (resolved at the API edge) wins;
    // when the client is silent the server-level
    // --default-chat-template-kwargs directive applies; when that too is
    // unspecified, MODEL.toml decides inside resolve_thinking.
    let mut thinking_directive = req.thinking;
    if !thinking_directive.is_explicit() {
        thinking_directive = state.default_thinking;
    }
    let gen_max =
        thinking::generation_max_tokens(req.max_tokens, tools_active, state.tool_max_tokens);
    let (enable_thinking, thinking_budget) =
        thinking::resolve_thinking(state, thinking_directive, gen_max as u32, tools_active);
    let reasoning_effort = effective_reasoning_effort(
        req.reasoning_effort,
        state.default_reasoning_effort,
        enable_thinking,
    );
    let us_thinking = _t_phase.elapsed().as_micros() - us_msg_entry;

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
        reasoning_effort,
        // Per-request chat_template_kwargs.preserve_thinking wins; the
        // MODEL.toml [behavior] override is the fallback; both unset
        // leaves the Jinja variable undefined (template default).
        req.preserve_thinking.or(state.behavior.preserve_thinking),
        tools_active,
    )?;
    if state.chat.phase_timing {
        let us_template = _t_phase.elapsed().as_micros() - us_msg_entry - us_thinking;
        tracing::info!(
            "CHAT_PHASE prepare: msg_entry={us_msg_entry}us thinking={us_thinking}us \
             template_render_and_tokenize={us_template}us prompt_tokens={}",
            prompt_tokens.len()
        );
    }

    Ok(PreparedChat {
        tools_active,
        cwd_hint,
        image_pixels,
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    })
}

/// The effort string handed to the template: the per-request value
/// wins, then the server-level default (`--default-chat-template-kwargs`
/// `{"reasoning_effort":...}`), and only while thinking is ON — a
/// thinking-off resolution (client "none", `--disable-thinking`,
/// `thinking_in_tools=false`, ...) must never carry an effort directive
/// into the render (Qwen3.8 gates its effort block on `enable_thinking`;
/// Mistral's validator raises on any effort other than `none` when the
/// settings line is emitted). `None` with thinking on falls through to
/// the cross-template `"medium"` fallback in `tokenizer/chat_render.rs`
/// — the neutral tier, NOT the pre-2026-08 `"high"` that Qwen3.8
/// escalated to its most expensive `xhigh` directive.
fn effective_reasoning_effort(
    request: Option<crate::ir::ReasoningEffort>,
    server_default: Option<crate::ir::ReasoningEffort>,
    enable_thinking: bool,
) -> Option<crate::ir::ReasoningEffort> {
    if enable_thinking {
        request.or(server_default)
    } else {
        None
    }
}

/// Resolve the parser contribution independently of the native template renderer.
pub(crate) fn parser_tool_prompt(
    parser: &dyn crate::tool_parser::ToolCallParser,
    tools: &[crate::tool_parser::ToolDefinition],
    choice: &crate::tool_parser::ToolChoice,
    levers: &crate::tool_parser::PromptLevers,
    native_qwen_template: bool,
) -> String {
    if native_qwen_template && !levers.tscg && matches!(parser.name(), "qwen3_coder" | "qwen3_xml")
    {
        // The checkpoint template already owns the schema and XML format.
        // Keep required/named tool-choice constraints, but do not inject a
        // second copy or unrelated Bash/Write/Edit behavioral instructions.
        let mut prompt = String::new();
        crate::tool_parser::append_tool_choice_instruction(&mut prompt, choice);
        return prompt;
    }
    parser.system_prompt(tools, choice, levers)
}

/// Inject a parser's behavioral prompt without changing requests for parsers
/// whose native chat template already owns tool instructions.
pub(crate) fn inject_tool_system_prompt(
    messages: &mut Vec<crate::ir::Message>,
    tool_prompt: String,
) {
    if tool_prompt.is_empty() {
        return;
    }

    if let Some(first) = messages
        .first_mut()
        .filter(|m| m.role == crate::ir::Role::System)
    {
        first.prepend_text(&format!("{tool_prompt}\n\n"));
    } else {
        messages.insert(0, crate::ir::Message::synthetic_system(tool_prompt));
    }
}

#[cfg(test)]
mod tests {
    use super::{effective_reasoning_effort, inject_tool_system_prompt};
    use crate::ir::{Message, ReasoningEffort, Role};

    /// The four-way contract of the template-effort resolution: request
    /// beats server default, the server default fills silence, and a
    /// thinking-off resolution suppresses BOTH — `disable_thinking` (and
    /// every other off-path) must fully bypass the effort directive
    /// (Qwen3.8's template gates on it; the BFCL gate serves
    /// thinking-off and depends on the bypass).
    #[test]
    fn effort_resolution_contract() {
        let low = Some(ReasoningEffort::Low);
        let max = Some(ReasoningEffort::Max);
        // Request wins over the server default.
        assert_eq!(effective_reasoning_effort(low, max, true), low);
        // Server default fills a silent request.
        assert_eq!(effective_reasoning_effort(None, max, true), max);
        // Fully unset: None → the "medium" chat_render fallback (the
        // neutral tier), asserted byte-level in tokenizer/tests/qwen_dense.rs.
        assert_eq!(effective_reasoning_effort(None, None, true), None);
        // Thinking off suppresses both sources.
        assert_eq!(effective_reasoning_effort(low, max, false), None);
        assert_eq!(effective_reasoning_effort(None, max, false), None);
    }

    #[test]
    fn empty_tool_prompt_does_not_change_existing_system_message() {
        let mut messages = vec![Message::synthetic_system("original".into())];
        let before = messages.clone();

        inject_tool_system_prompt(&mut messages, String::new());

        assert_eq!(messages, before);
    }

    #[test]
    fn empty_tool_prompt_does_not_insert_system_message() {
        let mut messages = vec![Message {
            role: Role::User,
            content: vec![crate::ir::ContentPart::Text("hello".into())],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
            tool_error: false,
        }];
        let before = messages.clone();

        inject_tool_system_prompt(&mut messages, String::new());

        assert_eq!(messages, before);
    }

    #[test]
    fn nonempty_tool_prompt_preserves_existing_injection_behavior() {
        let mut messages = vec![Message::synthetic_system("original".into())];

        inject_tool_system_prompt(&mut messages, "tool instructions".into());

        assert_eq!(messages[0].text(), "tool instructions\n\noriginal");
    }
}

#[cfg(test)]
#[path = "../../tokenizer/tests/native_tool_prompt.rs"]
mod native_tool_prompt;
