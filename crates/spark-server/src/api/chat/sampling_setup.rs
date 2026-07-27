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
use crate::ir::ChatRequest;
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
    req: &ChatRequest,
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
    // ATLAS_FORCE_TEMP_ZERO=1 — diagnostic override that forces fully greedy
    // deterministic decoding, ignoring client params AND MODEL.toml presets.
    // Used for layer-by-layer cosine comparison against vLLM (same env-var
    // contract on the vLLM side, VLLM_FORCE_TEMP_ZERO). At T=0 with identical
    // weights+tokens, two engines should produce bit-identical token streams;
    // any divergence localises a numerical bug.
    let force_temp_zero = std::env::var("ATLAS_FORCE_TEMP_ZERO")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // Core sampling (temp/top_k/top_p) defaults to the model's shipped
    // generation_config.json (state.default_*) — matching vLLM and the model
    // author's recommended config — instead of the hand-curated MODEL.toml
    // [sampling] presets. The lower preset temps (e.g. Holo thinking/tools =
    // 0.6 vs generation_config 1.0) made the model over-commit to tool calls
    // vs vLLM: lower temperature is more deterministic, so it picks "call the
    // tool" far more consistently. The selected preset still drives the
    // penalties below (generation_config doesn't define those); min_p and
    // top_n_sigma already default to generation_config.
    let temperature = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .temperature
            .unwrap_or(state.default_temperature)
    };
    let top_k = if force_temp_zero {
        0
    } else {
        req.sampling.top_k.unwrap_or(state.default_top_k)
    };
    let top_p = if force_temp_zero {
        1.0
    } else {
        req.sampling.top_p.unwrap_or(state.default_top_p)
    };
    let top_n_sigma = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .top_n_sigma
            .unwrap_or(state.default_top_n_sigma)
    };
    let min_p = if force_temp_zero {
        0.0
    } else {
        req.sampling.min_p.unwrap_or(state.default_min_p)
    };
    let repetition_penalty = if force_temp_zero {
        1.0
    } else {
        req.sampling
            .repetition_penalty
            .unwrap_or(preset.repetition_penalty)
    };
    let presence_penalty = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .presence_penalty
            .unwrap_or(preset.presence_penalty)
    };
    let frequency_penalty = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .frequency_penalty
            .unwrap_or(preset.frequency_penalty)
    };
    // Per-model server-side sampling SAFETY FLOOR/CEILING (MODEL.toml
    // [behavior]). Binds AFTER request/preset resolution so model stability
    // does NOT depend on the client volunteering safe params — the Claude-Code
    // loop fix (an unfloored min_p let the FP8/NVFP4 degenerate tail be sampled
    // into repetition loops; measured 2026-06-07: 0.05 → 4 watchdog fires
    // become 0). 0.0 = disabled (no-op). Skipped under force_temp_zero (that
    // diagnostic override deliberately drives greedy).
    let min_p = if !force_temp_zero && state.behavior.min_p_floor > 0.0 {
        min_p.max(state.behavior.min_p_floor)
    } else {
        min_p
    };
    let temperature = if !force_temp_zero && state.behavior.temperature_max > 0.0 {
        temperature.min(state.behavior.temperature_max)
    } else {
        temperature
    };
    let dry_multiplier = if force_temp_zero {
        0.0
    } else {
        preset.dry_multiplier
    };
    let dry_base = preset.dry_base;
    let dry_allowed_length = preset.dry_allowed_length;
    let lz_penalty = if force_temp_zero {
        0.0
    } else {
        preset.lz_penalty
    };

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

    // Logit bias (already parsed to typed pairs at the API edge).
    let mut logit_bias: Vec<(u32, f32)> = if force_temp_zero {
        Vec::new()
    } else {
        req.logit_bias.clone()
    };

    // Exponential `<tool_call>` bias decay. Skipped under ATLAS_FORCE_TEMP_ZERO
    // so the argmax is determined purely by raw logits (matches vLLM's path).
    if !force_temp_zero
        && tools_active
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
    //
    // #192: `</tool_call>` is deliberately NOT a stop token. It used to be
    // pushed here for every tools-active request, which (a) hard-stopped the
    // MTP/emit path at the FIRST closed tool call (the token hit the EOS
    // handler and was even dropped from the output), and (b) landed in the
    // grammar's stop-token exemption set, so the matcher never advanced
    // across the end-tag literal and desynced before a second call. vLLM
    // parity: generation continues past a closed call until natural EOS so
    // the model can emit parallel calls; the scheduler's tool watchdogs
    // (post-completion open cap, prose budget, loop detectors) bound run-on.
    let stop_tokens = tokenize_stop_sequences(&state.tokenizer, &req.stop);

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
    // The wire's `{"type":"text"}` was mapped to `None` at the edge, so
    // presence alone means a real constraint.
    let has_response_format = req.response_format.is_some();
    let tool_choice_none = req
        .tool_choice
        .as_ref()
        .is_some_and(|tc| matches!(tc, tool_parser::ToolChoice::Mode(m) if m == "none"));
    let response_format_only = has_response_format && (!tools_active || tool_choice_none);

    // Grammar spec (XGrammar structural-tag enforcement).
    let use_triggers = !tool_choice_required;
    let grammar_spec: Option<GrammarSpec> = if response_format_only {
        match req.response_format.as_ref().unwrap() {
            crate::ir::ResponseFormat::JsonObject => Some(GrammarSpec::JsonObject),
            crate::ir::ResponseFormat::JsonSchema { schema, .. } => Some(GrammarSpec::JsonSchema {
                schema: schema.to_string(),
            }),
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
        let mut tools = req.tools.clone();
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
    let timeout_secs = req.timeout_secs.unwrap_or(state.request_timeout as f32);
    let timeout_at = if timeout_secs > 0.0 {
        Some(std::time::Instant::now() + std::time::Duration::from_secs_f32(timeout_secs))
    } else {
        None
    };

    // Pre-resolved from the wire's logprobs/top_logprobs pair at the edge.
    let top_logprobs = req.top_logprobs;

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

// `resolve_top_logprobs` moved to the OpenAI edge (`openai/to_ir.rs`)
// — the envelope carries the already-resolved count.

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
