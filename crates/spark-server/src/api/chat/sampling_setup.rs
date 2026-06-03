// SPDX-License-Identifier: AGPL-3.0-only
//
// Sampling-preset selection, stop-token tokenisation, grammar-spec
// construction, and timeout / logprobs resolution.
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;
use crate::openai::ChatCompletionRequest;
use crate::tool_parser;

use super::super::compact::openai_error_response;
use super::super::inference_impl::tokenize_stop_sequences;
use super::super::inference_types::GrammarSpec;

pub(super) struct SamplingSetup {
    pub(super) temperature: f32,
    pub(super) top_k: u32,
    pub(super) top_p: f32,
    pub(super) top_n_sigma: f32,
    pub(super) min_p: f32,
    pub(super) repetition_penalty: f32,
    pub(super) presence_penalty: f32,
    pub(super) frequency_penalty: f32,
    pub(super) dry_multiplier: f32,
    pub(super) dry_base: f32,
    pub(super) dry_allowed_length: u32,
    pub(super) lz_penalty: f32,
    pub(super) logit_bias: Vec<(u32, f32)>,
    pub(super) max_tokens: usize,
    pub(super) stop_tokens: Vec<u32>,
    pub(super) tool_choice_required: bool,
    pub(super) grammar_spec: Option<GrammarSpec>,
    pub(super) timeout_at: Option<std::time::Instant>,
    pub(super) top_logprobs: Option<u8>,
}

fn tool_choice_required_for_parser(
    tools_active: bool,
    tool_choice: Option<&tool_parser::ToolChoice>,
    parser_name: Option<&str>,
) -> bool {
    if !tools_active {
        return false;
    }

    let explicit_required = tool_choice.is_some_and(|tc| {
        matches!(tc, tool_parser::ToolChoice::Mode(m) if m == "required")
            || matches!(tc, tool_parser::ToolChoice::Specific { .. })
    });
    let parser_required = matches!(parser_name, Some("minimax_xml"));

    explicit_required || parser_required
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(super) fn build_sampling(
    state: &Arc<AppState>,
    req: &ChatCompletionRequest,
    enable_thinking: bool,
    tools_active: bool,
    suppress_tool_call: bool,
    tool_call_repeat_count: usize,
) -> Result<SamplingSetup, Response> {
    // Preset selection.
    let preset = if tools_active {
        &state.sampling_presets.tools
    } else if enable_thinking {
        &state.sampling_presets.thinking_text
    } else {
        &state.sampling_presets.non_thinking
    };
    let temperature = req.temperature.unwrap_or(preset.temperature);
    let top_k = req.top_k.unwrap_or(preset.top_k);
    let top_p = req.top_p.unwrap_or(preset.top_p);
    let top_n_sigma = req.top_n_sigma.unwrap_or(state.default_top_n_sigma);
    let min_p = req.min_p.unwrap_or(state.default_min_p);
    let repetition_penalty = req.repetition_penalty.unwrap_or(preset.repetition_penalty);
    let presence_penalty = req.presence_penalty.unwrap_or(preset.presence_penalty);
    let frequency_penalty = req.frequency_penalty.unwrap_or(preset.frequency_penalty);
    let dry_multiplier = preset.dry_multiplier;
    let dry_base = preset.dry_base;
    let dry_allowed_length = preset.dry_allowed_length;
    let lz_penalty = preset.lz_penalty;

    // OpenAI-style penalty range validation.
    if !(-2.0..=2.0).contains(&presence_penalty) {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("presence_penalty must be between -2.0 and 2.0, got {presence_penalty}"),
        ));
    }
    if !(-2.0..=2.0).contains(&frequency_penalty) {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("frequency_penalty must be between -2.0 and 2.0, got {frequency_penalty}"),
        ));
    }

    // Logit bias from OpenAI (string keys) → Vec<(u32, f32)>.
    let mut logit_bias: Vec<(u32, f32)> = req.logit_bias.as_ref().map_or(Vec::new(), |map| {
        map.iter()
            .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
            .collect()
    });

    // Exponential `<tool_call>` bias decay.
    if tools_active
        && !suppress_tool_call
        && let Some(tc_id) = state.tool_call_start_token_id
    {
        let bias = match tool_call_repeat_count {
            0 | 1 => 3.0,
            2 => 0.0,
            3 => -5.0,
            _ => -10.0,
        };
        if bias != 0.0 {
            logit_bias.push((tc_id, bias));
        }
    }

    // max_tokens cap when tools are active.
    let max_tokens = if tools_active {
        let capped = req.max_tokens.min(state.tool_max_tokens);
        if capped < req.max_tokens {
            tracing::info!(
                "Tool max_tokens cap: {} → {} (tool_max_tokens={})",
                req.max_tokens,
                capped,
                state.tool_max_tokens
            );
        }
        capped
    } else {
        req.max_tokens
    };

    // Stop tokens.
    let mut stop_tokens = tokenize_stop_sequences(&state.tokenizer, &req.stop);
    if tools_active
        && let Ok(ids) = state.tokenizer.encode("</tool_call>")
        && ids.len() == 1
    {
        stop_tokens.push(ids[0]);
    }

    // Tool-choice + parser-driven required mode.
    let tool_choice_required = tool_choice_required_for_parser(
        tools_active,
        req.tool_choice.as_ref(),
        state.tool_call_parser.as_ref().map(|p| p.name()),
    );

    // response_format + tools coexistence.
    //
    // OpenAI's API allows both fields in the same request; agentic pipelines
    // routinely set both (the model emits a tool call on turn N, then a
    // schema-shaped final answer on turn N+1). XGrammar's structural-tag
    // grammar enforces *one* shape per request, so we pick which one wins:
    //   * `tool_choice="none"` → tools won't be called, enforce response_format
    //   * any other tool_choice → enforce tool-call grammar; the schema text
    //     is conventionally embedded in the user/system message by the
    //     caller, and capable models (Qwen3.6, etc.) follow it without
    //     server-side enforcement on free-text turns.
    let has_response_format = req
        .response_format
        .as_ref()
        .is_some_and(|rf| !matches!(rf, crate::openai::ResponseFormat::Text));
    let tool_choice_none = req
        .tool_choice
        .as_ref()
        .is_some_and(|tc| matches!(tc, tool_parser::ToolChoice::Mode(m) if m == "none"));
    let response_format_only = has_response_format && (!tools_active || tool_choice_none);

    // Grammar spec (XGrammar structural-tag enforcement).
    let use_triggers = !tool_choice_required;
    let grammar_spec: Option<GrammarSpec> = if response_format_only {
        match req.response_format.as_ref().unwrap() {
            crate::openai::ResponseFormat::JsonObject => Some(GrammarSpec::JsonObject),
            crate::openai::ResponseFormat::JsonSchema { json_schema } => {
                Some(GrammarSpec::JsonSchema {
                    schema: json_schema.schema.to_string(),
                })
            }
            crate::openai::ResponseFormat::Text => None,
        }
    } else if tools_active && state.behavior.disable_tool_grammar {
        // Structure-snowballing escape hatch (arXiv:2604.06066): this
        // model tool-calls more reliably unconstrained. Tool calls are
        // still parsed from the output — just not grammar-enforced.
        tracing::info!("MODEL.toml [behavior].disable_tool_grammar=true — tool-call grammar OFF");
        None
    } else if tools_active {
        if has_response_format {
            tracing::info!(
                "response_format + tools both set; enforcing tool-call grammar. \
                 Schema-shape compliance falls to the model (embed schema text in \
                 the user/system message for best results)."
            );
        }
        let parser = state.tool_call_parser.as_ref().map(std::sync::Arc::clone);
        let mut tools = req.tools.as_ref().cloned().unwrap_or_default();
        if let Some(tool_parser::ToolChoice::Specific { ref function }) = req.tool_choice {
            tools.retain(|t| t.function.name == function.name);
        }
        parser.map(|p| GrammarSpec::ToolCall {
            tools,
            parser: p,
            use_triggers,
        })
    } else {
        None
    };

    // Timeout deadline.
    let timeout_secs = req.timeout.unwrap_or(state.request_timeout as f32);
    let timeout_at = if timeout_secs > 0.0 {
        Some(std::time::Instant::now() + std::time::Duration::from_secs_f32(timeout_secs))
    } else {
        None
    };

    // top_logprobs (OpenAI spec: 0-20).
    let top_logprobs = req.top_logprobs.map(|n| n.min(20));

    Ok(SamplingSetup {
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        max_tokens,
        stop_tokens,
        tool_choice_required,
        grammar_spec,
        timeout_at,
        top_logprobs,
    })
}

#[cfg(test)]
mod tests {
    use super::tool_choice_required_for_parser;
    use crate::tool_parser::{ToolChoice, ToolChoiceFunction};

    #[test]
    fn bare_json_auto_uses_triggered_grammar() {
        assert!(!tool_choice_required_for_parser(
            true,
            None,
            Some("bare_json")
        ));
    }

    #[test]
    fn bare_json_required_mode_enforces_from_first_token() {
        let choice = ToolChoice::Mode("required".to_string());

        assert!(tool_choice_required_for_parser(
            true,
            Some(&choice),
            Some("bare_json")
        ));
    }

    #[test]
    fn specific_function_enforces_from_first_token() {
        let choice = ToolChoice::Specific {
            function: ToolChoiceFunction {
                name: "memory".to_string(),
            },
        };

        assert!(tool_choice_required_for_parser(
            true,
            Some(&choice),
            Some("bare_json")
        ));
    }

    #[test]
    fn minimax_xml_remains_parser_required() {
        assert!(tool_choice_required_for_parser(
            true,
            None,
            Some("minimax_xml")
        ));
    }

    #[test]
    fn inactive_tools_are_not_required() {
        assert!(!tool_choice_required_for_parser(
            false,
            None,
            Some("bare_json")
        ));
    }
}
