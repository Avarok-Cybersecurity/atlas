// SPDX-License-Identifier: AGPL-3.0-only

//! Axum handlers for the llama.cpp routes that are thin over a pure module:
//! `GET /props`, `POST /apply-template`, and the `/embedding` refusal. The
//! parsing and the response bodies are in the sibling modules and are tested
//! there; these only fetch the model state.

use axum::body::Bytes;
use axum::response::{IntoResponse, Json, Response};

use crate::main_modules::model_host::CurrentModel;

use super::apply_template::{apply_template_response, parse_apply_template};
use super::error::{CompatError, parse_json_object};
use super::props::{PropsInputs, SamplingDefaults, build_info, props_json};

/// The sampling defaults a request that omits a field gets.
pub(crate) fn sampling_defaults(state: &crate::AppState) -> SamplingDefaults {
    SamplingDefaults {
        temperature: state.default_temperature,
        top_k: state.default_top_k,
        top_p: state.default_top_p,
        min_p: state.default_min_p,
        top_n_sigma: state.default_top_n_sigma,
        repetition_penalty: state.sampling_presets.non_thinking.repetition_penalty,
    }
}

/// `GET /props`.
pub async fn props(CurrentModel(state): CurrentModel) -> Response {
    let chat_template = state.tokenizer.chat_template_source().to_string();
    let bos = state.tokenizer.bos_token_text();
    let eos = state.tokenizer.eos_token_text();
    let build = build_info();
    let body = props_json(&PropsInputs {
        model_name: &state.model_name,
        n_ctx: state.max_seq_len,
        total_slots: state.max_num_seqs,
        chat_template: &chat_template,
        bos_token: &bos,
        eos_token: &eos,
        build_info: &build,
        vision: state.vision_config.is_some(),
        defaults: sampling_defaults(&state),
    });
    Json(body).into_response()
}

/// `POST /apply-template`.
pub async fn apply_template(CurrentModel(state): CurrentModel, body: Bytes) -> Response {
    let run = || -> Result<serde_json::Value, CompatError> {
        let map = parse_json_object(&body)?;
        let req = parse_apply_template(&map)?;
        // Thinking follows the server's resolved default for the model unless
        // the request's `chat_template_kwargs` say otherwise; the server-wide
        // `--disable-thinking` switch outranks both, as it does for chat.
        let enable_thinking = !state.disable_thinking
            && req
                .enable_thinking
                .unwrap_or(state.behavior.thinking_default);
        let prompt = state
            .tokenizer
            .render_chat_text_with_effort(
                &req.messages,
                req.tools.as_deref(),
                enable_thinking,
                state.behavior.disable_tool_steering,
                req.reasoning_effort.as_deref(),
                state.behavior.preserve_thinking,
            )
            .map_err(|e| CompatError::invalid(format!("chat template error: {e}")))?;
        Ok(apply_template_response(prompt))
    };
    match run() {
        Ok(v) => Json(v).into_response(),
        Err(e) => e.into_response(),
    }
}

/// `POST /embedding` and `POST /embeddings`: Atlas serves generative models
/// only. llama.cpp answers 501 `not_supported_error` when it was not started
/// with embeddings, and Atlas answers the same way, so a client that probes
/// for the feature falls back instead of retrying.
pub async fn embedding_unsupported() -> Response {
    CompatError::not_implemented(
        "This server does not support embeddings. Atlas serves generative \
         (chat and completion) models only.",
    )
    .into_response()
}
