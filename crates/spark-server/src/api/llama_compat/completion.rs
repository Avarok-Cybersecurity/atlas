// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /completion`: llama.cpp's native completion route.
//!
//! No second inference path: the request is translated into the same
//! `InferenceRequest::Streaming` the OpenAI `/v1/completions` streaming route
//! builds (through the same builder), and sent to the same scheduler. Both the
//! streamed and the blocking answer are produced from the token events, so the
//! cancel flag can end generation the moment a stop string completes, which the
//! scheduler's blocking request has no way to do.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::body::Bytes;
use axum::response::{IntoResponse, Json, Response};
use tokio::sync::mpsc;

use crate::AppState;
use crate::api::completions::build_streaming_request;
use crate::api::completions_exec::CompletionParams;
use crate::api::{GrammarSpec, InferenceRequest, StreamEvent};
use crate::main_modules::model_host::CurrentModel;
use crate::openai::CompletionRequest;

use super::codec::TextCodec;
use super::completion_stream::stream_response;
use super::constraint::Constraint;
use super::error::{CompatError, parse_json_object};
use super::finish::{DoneInfo, final_value};
use super::generation::Generation;
use super::handlers::sampling_defaults;
use super::params::{CompletionSpec, ParseContext, parse_completion};

/// The tokenizer behind an `Arc<AppState>`, as an owned `Arc<dyn TextCodec>`
/// that a streaming response can carry.
pub(crate) struct StateCodec(pub(crate) Arc<AppState>);

impl TextCodec for StateCodec {
    fn encode_text(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
        self.0.tokenizer.encode_text(text, add_special)
    }
    fn decode_text(&self, ids: &[u32], skip_special: bool) -> Result<String, String> {
        self.0.tokenizer.decode_text(ids, skip_special)
    }
    fn vocab_size(&self) -> usize {
        self.0.tokenizer.vocab_size()
    }
    fn token_bytes(&self, id: u32) -> Vec<u8> {
        self.0.tokenizer.token_bytes(id)
    }
    fn decode_incremental(&self, ids: &[u32], prefix: &mut usize, read: &mut usize) -> String {
        self.0.tokenizer.decode_incremental(ids, prefix, read)
    }
}

/// A request that is queued and producing events.
pub(crate) struct Started {
    pub spec: CompletionSpec,
    pub rx: mpsc::Receiver<StreamEvent>,
    /// Set to end generation early (a stop string completed).
    pub cancel: Arc<AtomicBool>,
}

/// Map a validated request onto the OpenAI-completions request the shared
/// builder reads `max_tokens`, `seed`, `min_tokens` and `echo` from.
fn to_completion_request(
    model: &str,
    spec: &CompletionSpec,
) -> Result<CompletionRequest, CompatError> {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "prompt": "",
        "max_tokens": spec.max_tokens,
        "seed": spec.seed,
        "stream": true,
    }))
    .map_err(|e| CompatError::server(format!("internal request mapping failed: {e}")))
}

fn grammar_spec_of(c: &Constraint) -> GrammarSpec {
    match c {
        Constraint::JsonSchema(s) => GrammarSpec::JsonSchema { schema: s.clone() },
        Constraint::Ebnf(g) => GrammarSpec::Ebnf { grammar: g.clone() },
    }
}

/// Build the scheduler request and the channel its events arrive on.
///
/// Stop strings are NOT passed as scheduler stop tokens: a single-token stop
/// would end the sequence without the text ever reaching the matcher, and the
/// response could not say `stop_type: "word"`. The matcher in [`Generation`]
/// sees every stop and cancels the request instead.
pub(crate) fn build_request(
    model: &str,
    spec: &CompletionSpec,
    adapter_slot: i32,
    timeout_at: Option<std::time::Instant>,
) -> Result<
    (
        InferenceRequest,
        Arc<AtomicBool>,
        mpsc::Receiver<StreamEvent>,
    ),
    CompatError,
> {
    let req = to_completion_request(model, spec)?;
    let params = CompletionParams {
        temperature: spec.temperature,
        top_k: spec.top_k,
        top_p: spec.top_p,
        top_n_sigma: spec.top_n_sigma,
        min_p: spec.min_p,
        repetition_penalty: spec.repetition_penalty,
        presence_penalty: spec.presence_penalty,
        frequency_penalty: spec.frequency_penalty,
        logit_bias: spec.logit_bias.clone(),
        stop_tokens: Vec::new(),
        repetition_detection: None,
        logprobs_k: (spec.n_probs > 0).then_some(spec.n_probs),
        adapter_slot,
        src_lang_id: 0,
        tgt_lang_id: 0,
        num_beams: 1,
        length_penalty: 1.0,
        early_stopping: false,
        grammar_spec: spec.constraint.as_ref().map(grammar_spec_of),
    };
    // Same channel size as the other streaming handlers.
    let (token_tx, rx) = mpsc::channel::<StreamEvent>(1024);
    let session_hash = crate::session_manager::compute_session_hash(&spec.prompt_tokens);
    let mut request = build_streaming_request(
        &req,
        params,
        Arc::new(spec.prompt_tokens.clone()),
        session_hash,
        timeout_at,
        token_tx,
    );
    let InferenceRequest::Streaming {
        cancel_flag,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        ..
    } = &mut request
    else {
        return Err(CompatError::server(
            "streaming request builder changed shape",
        ));
    };
    // The shared builder fixes DRY off (it is the OpenAI route's choice);
    // llama.cpp's request sets it.
    *dry_multiplier = spec.dry_multiplier;
    *dry_base = spec.dry_base;
    *dry_allowed_length = spec.dry_allowed_length;
    let cancel = Arc::clone(cancel_flag);
    Ok((request, cancel, rx))
}

/// Parse, validate and queue a `/completion` request.
async fn start(state: &Arc<AppState>, body: &[u8]) -> Result<Started, Response> {
    let to_resp = |e: CompatError| e.into_response();
    let map = parse_json_object(body).map_err(to_resp)?;
    let codec = StateCodec(Arc::clone(state));
    let spec = parse_completion(
        &map,
        &ParseContext {
            codec: &codec,
            defaults: sampling_defaults(state),
            max_seq_len: state.max_seq_len,
            grammar_available: state.grammar_available,
        },
    )
    .map_err(to_resp)?;
    let adapter_slot =
        crate::api::lora_control::resolve_request_adapter_slot(state, None, &state.model_name)
            .await?;
    let (request, cancel, rx) = build_request(
        &state.model_name,
        &spec,
        adapter_slot,
        state.request_deadline(None),
    )
    .map_err(to_resp)?;
    state
        .request_tx
        .send(request)
        .await
        .map_err(|_| to_resp(CompatError::unavailable("Scheduler queue full")))?;
    Ok(Started { spec, rx, cancel })
}

/// Collect every event into one response body.
async fn collect(state: Arc<AppState>, started: Started) -> Result<Response, CompatError> {
    let Started {
        spec,
        mut rx,
        cancel,
    } = started;
    let codec: Arc<dyn TextCodec> = Arc::new(StateCodec(Arc::clone(&state)));
    let mut generation = Generation::new(Arc::clone(&codec), &spec.stop, spec.n_probs);
    let mut done = None;
    while let Some(event) = rx.recv().await {
        let (token, lp) = match event {
            StreamEvent::Token(t) => (t, None),
            StreamEvent::TokenWithLogprobs(t, lp) => (t, Some(lp)),
            StreamEvent::PromptLogprobs(_) => continue,
            StreamEvent::Done {
                finish_reason,
                completion_tokens,
                time_to_first_token_ms,
                decode_time_ms,
                cached_prompt_tokens,
                ..
            } => {
                done = Some(DoneInfo {
                    finish_reason,
                    completion_tokens,
                    ttft_ms: time_to_first_token_ms,
                    decode_ms: decode_time_ms,
                    cached_prompt_tokens: cached_prompt_tokens as usize,
                });
                break;
            }
            StreamEvent::Error(msg) => {
                return Err(CompatError::server(format!("Inference error: {msg}")));
            }
        };
        // After a stop string, the scheduler may still deliver a few tokens
        // before it sees the cancel flag; they are dropped.
        if generation.stop_hit().is_none() && generation.push(token, lp.as_ref()).stopped {
            cancel.store(true, Ordering::Release);
        }
    }
    let Some(done) = done else {
        return Err(CompatError::server(
            "Inference ended without a result (the request was cancelled)",
        ));
    };
    if generation.stop_hit().is_none() {
        generation.finish();
    }
    let body = final_value(
        codec.as_ref(),
        &state.model_name,
        state.max_seq_len,
        &spec,
        &generation,
        &done,
    );
    Ok(Json(body).into_response())
}

/// `POST /completion`.
pub async fn completion(CurrentModel(state): CurrentModel, body: Bytes) -> Response {
    let started = match start(&state, &body).await {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    if started.spec.stream {
        stream_response(state, started)
    } else {
        collect(state, started)
            .await
            .unwrap_or_else(IntoResponse::into_response)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::props::SamplingDefaults;
    use super::super::toy_codec::ToyCodec;
    use super::*;

    fn spec(body: serde_json::Value) -> CompletionSpec {
        let ctx = ParseContext {
            codec: &ToyCodec,
            defaults: SamplingDefaults {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.8,
                min_p: 0.0,
                top_n_sigma: 0.0,
                repetition_penalty: 1.0,
            },
            max_seq_len: 64,
            grammar_available: true,
        };
        parse_completion(body.as_object().unwrap(), &ctx).expect("valid")
    }

    #[test]
    fn the_request_carries_budget_seed_and_prompt_to_the_scheduler() {
        let s = spec(json!({"prompt": "hi", "n_predict": 7, "seed": 5}));
        let (req, _cancel, _rx) = build_request("m", &s, 2, None).unwrap();
        assert_eq!(req.max_tokens(), 7);
        assert_eq!(req.seed(), Some(5));
        assert_eq!(req.adapter_slot(), 2);
        assert_eq!(req.prompt_len(), s.prompt_tokens.len());
        assert_eq!(req.top_k(), 20);
        assert_eq!(req.min_tokens(), 0);
    }

    #[test]
    fn stop_strings_never_become_scheduler_stop_tokens() {
        let s = spec(json!({"prompt": "hi", "stop": ["a", "END"]}));
        let (mut req, _c, _rx) = build_request("m", &s, -1, None).unwrap();
        assert!(req.take_stop_tokens().is_empty());
    }

    #[test]
    fn n_probs_turns_on_scheduler_logprobs() {
        let off = spec(json!({"prompt": "hi"}));
        assert_eq!(
            build_request("m", &off, -1, None).unwrap().0.top_logprobs(),
            None
        );
        let on = spec(json!({"prompt": "hi", "n_probs": 4}));
        assert_eq!(
            build_request("m", &on, -1, None).unwrap().0.top_logprobs(),
            Some(4)
        );
    }

    #[test]
    fn dry_settings_override_the_shared_builders_off_switch() {
        let s = spec(
            json!({"prompt": "hi", "dry_multiplier": 0.8, "dry_base": 2.0, "dry_allowed_length": 3}),
        );
        let (req, _c, _rx) = build_request("m", &s, -1, None).unwrap();
        assert_eq!(req.dry_multiplier(), 0.8);
        assert_eq!(req.dry_base(), 2.0);
        assert_eq!(req.dry_allowed_length(), 3);
        let off = spec(json!({"prompt": "hi"}));
        assert_eq!(
            build_request("m", &off, -1, None)
                .unwrap()
                .0
                .dry_multiplier(),
            0.0
        );
    }

    #[test]
    fn a_constraint_reaches_the_scheduler_as_a_grammar_spec() {
        let s = spec(json!({"prompt": "hi", "json_schema": {"type": "object"}}));
        let (mut req, _c, _rx) = build_request("m", &s, -1, None).unwrap();
        assert!(matches!(
            req.take_grammar_spec(),
            Some(GrammarSpec::JsonSchema { .. })
        ));
        let s = spec(json!({"prompt": "hi", "grammar": "root ::= \"a\""}));
        let (mut req, _c, _rx) = build_request("m", &s, -1, None).unwrap();
        assert!(matches!(
            req.take_grammar_spec(),
            Some(GrammarSpec::Ebnf { .. })
        ));
        let s = spec(json!({"prompt": "hi"}));
        let (mut req, _c, _rx) = build_request("m", &s, -1, None).unwrap();
        assert!(req.take_grammar_spec().is_none());
    }

    #[test]
    fn the_cancel_flag_is_the_one_the_scheduler_watches() {
        let s = spec(json!({"prompt": "hi"}));
        let (req, cancel, _rx) = build_request("m", &s, -1, None).unwrap();
        let InferenceRequest::Streaming { cancel_flag, .. } = &req else {
            panic!("expected a streaming request");
        };
        assert!(Arc::ptr_eq(cancel_flag, &cancel));
    }
}
