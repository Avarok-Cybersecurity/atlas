// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use crate::main_modules::model_host::CurrentModel;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use super::chat_stream::run_chat_stream;
use super::responses_stream::responses_endpoint_stream;
use super::responses_translate::{
    build_responses_usage, emit, find_frame_end, translate_chat_response_to_responses,
};
use super::stored::assistant_incoming_from_ir;
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
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};
use super::strip::strip_thinking_tags;

// Re-export sibling helpers via crate::api::* for short paths.
use super::inference_types::*;
use super::sanitizer::*;

pub async fn cancel_response(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    openai_error_response_with_param(
        StatusCode::BAD_REQUEST,
        format!(
            "Response '{id}' cannot be cancelled: Atlas completes responses synchronously. Cancel only applies when the request was created with `background: true`, which this server does not support."
        ),
        Some("id"),
        Some("response_not_cancellable"),
    )
}

/// GET /metrics — Prometheus metrics endpoint.
pub async fn metrics_handler() -> impl IntoResponse {
    use prometheus::Encoder;
    use std::fmt::Write;

    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    let mut text = String::from_utf8(buffer).unwrap_or_default();

    // Prefix cache counters (global atomics from spark-runtime)
    let hits = spark_runtime::prefix_cache::cache_hit_count();
    let misses = spark_runtime::prefix_cache::cache_miss_count();
    let hit_tokens = spark_runtime::prefix_cache::cache_hit_tokens_total();
    let total = hits + misses;
    let hit_rate = if total > 0 {
        hits as f64 / total as f64
    } else {
        0.0
    };

    let _ = write!(
        text,
        "\
        # HELP atlas_prefix_cache_hits_total Prefix cache lookups that found cached blocks\n\
        # TYPE atlas_prefix_cache_hits_total counter\n\
        atlas_prefix_cache_hits_total {hits}\n\
        # HELP atlas_prefix_cache_misses_total Prefix cache lookups with no match\n\
        # TYPE atlas_prefix_cache_misses_total counter\n\
        atlas_prefix_cache_misses_total {misses}\n\
        # HELP atlas_prefix_cache_hit_tokens_total Tokens reused from prefix cache\n\
        # TYPE atlas_prefix_cache_hit_tokens_total counter\n\
        atlas_prefix_cache_hit_tokens_total {hit_tokens}\n\
        # HELP atlas_prefix_cache_hit_rate Prefix cache hit rate (0-1)\n\
        # TYPE atlas_prefix_cache_hit_rate gauge\n\
        atlas_prefix_cache_hit_rate {hit_rate:.4}\n"
    );

    // Entropy monitoring (global atomics from spark-runtime sampler)
    let entropy = spark_runtime::sampler::last_entropy();
    let low_entropy = spark_runtime::sampler::low_entropy_token_count();
    let total_sampled = spark_runtime::sampler::total_sampled_token_count();
    let low_ratio = if total_sampled > 0 {
        low_entropy as f64 / total_sampled as f64
    } else {
        0.0
    };

    // Kernel-resolution health. A gate can assert `== 0` instead of grepping
    // the boot log — which is the only way this stays checkable once the log
    // has rolled. Non-zero means some dispatch is on a silent fallback path.
    let unresolved = spark_runtime::kernel_audit::unresolved_lookups();
    let _ = write!(
        text,
        "\
        # HELP atlas_kernel_lookups_unresolved Kernel lookups that did not resolve for the live model\n\
        # TYPE atlas_kernel_lookups_unresolved gauge\n\
        atlas_kernel_lookups_unresolved {unresolved}\n"
    );

    // In-process device memory from the allocation ledger. The device-wide
    // figure (nvidia-smi) also holds the CUDA context, CUDA Graph pools and
    // library workspaces, which never pass through the ledger; reading both
    // separates what Atlas allocated from what the driver did.
    let (ledger_live, ledger_peak) = spark_runtime::run_metrics::ledger_bytes();
    let _ = write!(
        text,
        "\
        # HELP atlas_gpu_ledger_live_bytes Device bytes live in the Atlas allocation ledger\n\
        # TYPE atlas_gpu_ledger_live_bytes gauge\n\
        atlas_gpu_ledger_live_bytes {ledger_live}\n\
        # HELP atlas_gpu_ledger_peak_bytes High-water mark of the allocation ledger since the model loaded\n\
        # TYPE atlas_gpu_ledger_peak_bytes gauge\n\
        atlas_gpu_ledger_peak_bytes {ledger_peak}\n"
    );

    let _ = write!(
        text,
        "\
        # HELP atlas_token_entropy_last Most recent per-token entropy (nats)\n\
        # TYPE atlas_token_entropy_last gauge\n\
        atlas_token_entropy_last {entropy:.4}\n\
        # HELP atlas_low_entropy_tokens_total Tokens with entropy below 0.3\n\
        # TYPE atlas_low_entropy_tokens_total counter\n\
        atlas_low_entropy_tokens_total {low_entropy}\n\
        # HELP atlas_low_entropy_ratio Fraction of tokens with entropy below 0.3\n\
        # TYPE atlas_low_entropy_ratio gauge\n\
        atlas_low_entropy_ratio {low_ratio:.4}\n"
    );

    // The port sprint's (`port/consumer-opt`) series (`api/port_metrics.rs`).
    super::port_metrics::render(&mut text);

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        text,
    )
}

/// The readiness verdict, as a pure function of the two things that decide it.
///
/// Split out from `health` so both branches are testable without an axum
/// state, and because the precedence is the whole point: a **fault outranks a
/// published model**. Issue #429 was precisely a server reporting `ready` off
/// a published model while its CUDA context was destroyed — every request it
/// accepted could only 500. "A model is loaded" and "requests can succeed"
/// stopped being the same claim at that moment, and readiness means the
/// second.
pub(crate) fn readiness(
    model: Option<&str>,
    fault: Option<&str>,
) -> (StatusCode, serde_json::Value) {
    use super::llama_compat::error::{ErrorKind, error_object};
    // The 503 bodies also carry llama.cpp's `error` object, so a client that
    // reads `error.message` / `error.type` from llama-server sees the same
    // thing here; `status` and `reason` are Atlas's own and stay.
    let with_error = |status: &str, reason: Option<&str>, message: &str| {
        let mut body = error_object(
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorKind::Unavailable,
            message,
        );
        body["status"] = serde_json::json!(status);
        if let Some(r) = reason {
            body["reason"] = serde_json::json!(r);
        }
        body
    };
    if let Some(reason) = fault {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            with_error("faulted", Some(reason), &format!("GPU fault: {reason}")),
        );
    }
    match model {
        // llama-server answers `{"status": "ok"}`; clients compare against it.
        Some(name) => (
            StatusCode::OK,
            serde_json::json!({"status": "ok", "model": name}),
        ),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            with_error("loading", None, "Loading model"),
        ),
    }
}

/// GET /health — readiness probe (503 while model is loading). The healthy
/// answer is `{"status":"ok","model":...}`: `ok` is llama-server's status,
/// which llama.cpp clients compare against.
pub async fn health(
    State(host): State<Arc<crate::main_modules::model_host::ModelHost>>,
) -> Response {
    // Takes the HOST, not the model: reporting "no model" is the whole point of
    // this endpoint, so requiring one would make it unanswerable in exactly the
    // state it exists to describe.
    // A published model IS a ready one: the scheduler is running before the
    // state is published, and the listener does not bind until after. A second
    // readiness flag alongside it was a duplicate source of truth, and a stale
    // one — the swap published a new model while the router still held the
    // ORIGINAL flag, so /health reported "loading" forever after the first
    // swap.
    let state = host.current();
    let (code, body) = readiness(
        state.as_ref().map(|s| s.model_name.as_str()),
        avarok_core::fault::global().fault(),
    );
    (code, Json(body)).into_response()
}

/// GET /health/live — liveness probe. 200 normally; 503 once the GPU fault
/// latch is set.
///
/// Liveness deliberately fails on a fault, because the only remedy is a new
/// process: a destroyed CUDA context cannot be rebuilt in-place, so a
/// supervisor restarting this one is the correct response and a supervisor
/// leaving it running is not. The server also asks itself to shut down when
/// the latch trips (see the scheduler); this endpoint is what tells an
/// orchestrator the truth during the drain window, and what covers the case
/// where the drain cannot finish because in-flight work is stuck on the dead
/// context.
pub async fn health_live() -> Response {
    match avarok_core::fault::global().fault() {
        None => "ok".into_response(),
        Some(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "faulted", "reason": reason})),
        )
            .into_response(),
    }
}

/// GET /hardware — the serving box's hardware fingerprint, for benchmark
/// provenance. Probed on request (the sm-clock reading must be live), via
/// `spawn_blocking` because the vendor tools are synchronous subprocesses.
pub async fn hardware() -> Response {
    let hw = tokio::task::spawn_blocking(avarok_plugin::hardware::Hardware::probe)
        .await
        .unwrap_or_else(|_| avarok_plugin::hardware::Hardware::unknown());
    Json(hw).into_response()
}

/// GET /serve-config — the digests that say WHICH server this is: the bytes
/// of its binary and the arguments it was started with. Digests, never the
/// arguments (an argv can carry `--auth-token`). A benchmark that would
/// rather reuse a running server than start its own compares these against
/// what it would have started — see `cli::bench_lease`.
pub async fn serve_config() -> Response {
    let id =
        tokio::task::spawn_blocking(|| avarok_plugin::serve_identity::this_process().clone()).await;
    match id {
        Ok(id) => Json(id).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{e}")})),
        )
            .into_response(),
    }
}

/// POST /tokenize — two dialects on one route.
///
/// A body with `content` is llama.cpp's request (`content`, `add_special`,
/// `with_pieces`) and gets llama.cpp's answer (`{"tokens": [...]}`). Anything
/// else is Atlas's own (`prompt` or `messages`, answered with
/// `{tokens, count}`). The body is read as bytes so a missing
/// `Content-Type: application/json` is tolerated, as in llama-server.
pub async fn tokenize(CurrentModel(state): CurrentModel, body: axum::body::Bytes) -> Response {
    let value: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    if let Some(map) = value.as_object()
        && super::llama_compat::tokenize::is_llama_shape(map)
    {
        let run = || {
            let req = super::llama_compat::tokenize::parse_tokenize(map)?;
            super::llama_compat::tokenize::tokenize_response(&state.tokenizer, &req)
        };
        return match run() {
            Ok(v) => Json(v).into_response(),
            Err(e) => e.into_response(),
        };
    }
    let req: crate::openai::TokenizeRequest = match serde_json::from_value(value) {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    let tokens = if let Some(ref prompt) = req.prompt {
        match state.tokenizer.encode(prompt) {
            Ok(t) => t,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Tokenization error: {e}"),
                );
            }
        }
    } else if let Some(ref messages) = req.messages {
        let json_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content.text}))
            .collect();
        match state.tokenizer.apply_chat_template_jinja_with_effort(
            &json_messages,
            None,
            false,
            state.behavior.disable_tool_steering,
            None,
            // Honor the MODEL.toml preserve_thinking override so the counted
            // bytes match what serving renders (Qwen3.8 emits think markers
            // on historical assistant turns unless this is false).
            state.behavior.preserve_thinking,
        ) {
            Ok(t) => t,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Tokenization error: {e}"),
                );
            }
        }
    } else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "Either 'prompt' or 'messages' is required".to_string(),
        );
    };

    let count = tokens.len();
    Json(crate::openai::TokenizeResponse { tokens, count }).into_response()
}

/// Request body for POST /detokenize.
#[derive(serde::Deserialize)]
pub struct DetokenizeRequest {
    tokens: Vec<u32>,
}

/// POST /detokenize — decode token IDs back to text.
///
/// The request is the same in both dialects (`tokens`). The answer carries
/// llama.cpp's `content` (special tokens written out) and Atlas's own `text`
/// (special tokens dropped), so either kind of client finds its field.
pub async fn detokenize(CurrentModel(state): CurrentModel, body: axum::body::Bytes) -> Response {
    let req: DetokenizeRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    match super::llama_compat::tokenize::detokenize_response(&state.tokenizer, &req.tokens) {
        Ok(v) => Json(v).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e.message})),
        )
            .into_response(),
    }
}
