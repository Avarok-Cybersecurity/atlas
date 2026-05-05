// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 9-11 of `serve()`: build the axum router with CORS +
//! middleware, mark ready, bind the listener, and start the HTTP
//! server. Extracted (refactor wave-4e) for the ≤500 LoC cap.

use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use axum::routing::{get, post};

use crate::anthropic;
use crate::api;
use crate::main_modules::AppState;
use crate::main_modules::middleware::{
    openai_observability_middleware, rate_limit_middleware, require_auth_middleware,
};

pub(crate) async fn build_and_serve(
    state: Arc<AppState>,
    model_ready: Arc<std::sync::atomic::AtomicBool>,
    bind: &str,
    port: u16,
) -> Result<()> {
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers(tower_http::cors::Any);

    // Catch any panic in a handler and convert it to a 500 instead of
    // hanging the connection. With ~500 production unwraps still in the
    // codebase post-audit, this is cheap insurance — the panicking task
    // dies cleanly and the client sees a JSON error rather than a hung
    // socket. Default `tower_http::catch_panic` body is a plain text
    // "Service Internal Server Error"; we don't override the body so as
    // to avoid leaking backtrace contents to the client.
    let catch_panic = tower_http::catch_panic::CatchPanicLayer::new();

    let app = Router::new()
        .route("/v1/chat/completions", post(api::chat_completions))
        .route("/v1/chat/completions/{id}", get(api::get_stored_completion))
        .route("/v1/completions", post(api::completions))
        .route("/v1/responses", post(api::responses_endpoint))
        .route(
            "/v1/responses/{id}",
            get(api::get_stored_response).delete(api::delete_stored_response),
        )
        .route(
            "/v1/responses/{id}/input_items",
            get(api::list_response_input_items),
        )
        .route("/v1/responses/{id}/cancel", post(api::cancel_response))
        .route("/v1/conversations", post(api::create_conversation))
        .route(
            "/v1/conversations/{id}",
            get(api::get_conversation)
                .post(api::update_conversation)
                .delete(api::delete_conversation),
        )
        .route(
            "/v1/conversations/{id}/items",
            post(api::add_conversation_items).get(api::list_conversation_items),
        )
        .route(
            "/v1/conversations/{id}/items/{item_id}",
            get(api::get_conversation_item).delete(api::delete_conversation_item),
        )
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/v1/models", get(api::list_models))
        .route("/v1/models/{*model_id}", get(api::get_model))
        .route("/v1/embeddings", post(api::embeddings_stub))
        // 501 stubs: return an OpenAI-shaped error body so auto-probe
        // clients (Helicone, LangChain, Vercel AI SDK) fall back instead
        // of hanging on a silent 404.
        .route(
            "/v1/batches",
            post(api::batches_stub).get(api::batch_list_stub),
        )
        .route(
            "/v1/batches/{id}",
            get(api::batch_get_stub).delete(api::batch_get_stub),
        )
        .route("/v1/batches/{id}/cancel", post(api::batch_get_stub))
        .route("/v1/files", post(api::files_stub).get(api::files_stub))
        .route(
            "/v1/files/{id}",
            get(api::files_stub).delete(api::files_stub),
        )
        .route("/v1/files/{id}/content", get(api::files_stub))
        .route("/v1/audio/transcriptions", post(api::audio_stub))
        .route("/v1/audio/translations", post(api::audio_stub))
        .route("/v1/audio/speech", post(api::audio_stub))
        .route("/v1/images/generations", post(api::images_stub))
        .route("/v1/images/edits", post(api::images_stub))
        .route("/v1/images/variations", post(api::images_stub))
        .route("/v1/moderations", post(api::moderations_stub))
        .route("/tokenize", post(api::tokenize))
        .route("/detokenize", post(api::detokenize))
        .route("/health", get(api::health))
        .route("/health/live", get(api::health_live))
        .route("/metrics", get(api::metrics_handler))
        // Body size limit. Default 32 MB covers typical multi-image and
        // long-prompt requests; raise via `ATLAS_MAX_BODY_BYTES` (in
        // bytes) for unusual deployments. Lowering it protects against
        // DoS attempts that send oversized payloads to burn CPU on JSON
        // parsing + tokenization before the model even sees them.
        .layer(axum::extract::DefaultBodyLimit::max(
            std::env::var("ATLAS_MAX_BODY_BYTES")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(32 * 1024 * 1024),
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_auth_middleware,
        ))
        .layer(axum::middleware::from_fn(openai_observability_middleware))
        .layer(cors)
        .layer(catch_panic)
        .with_state(state);

    // Model loaded, scheduler running — mark as ready.
    model_ready.store(true, std::sync::atomic::Ordering::Relaxed);

    let addr = format!("{bind}:{port}");
    if bind == "0.0.0.0" {
        tracing::warn!(
            "Atlas is listening on {addr} — reachable from any host on the network. \
             If this machine is on a shared LAN or has a public IP, pass \
             --bind 127.0.0.1 (or set --require-auth and a real firewall) before \
             accepting traffic."
        );
    }
    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    // `into_make_service_with_connect_info` exposes the socket peer addr
    // to extractors — needed by `rate_limit_middleware` when the caller
    // didn't send X-Forwarded-For.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await?;

    Ok(())
}
