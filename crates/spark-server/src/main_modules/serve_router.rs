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
        .route("/v1/lora/active", post(api::set_active_lora))
        .route("/v1/lora/load", post(api::load_lora_into_slot))
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
    } else if bind == "127.0.0.1" || bind == "localhost" || bind == "::1" {
        // m00ch13 (Discord 2026-05-07): combined `--network host` with `-p 8000`
        // expecting LAN reachability and got refused from another machine. The
        // default loopback bind is correct for security, but the failure mode
        // ("connection refused from $LAN_IP") is opaque without this hint.
        tracing::info!(
            "API reachable only from this machine (loopback). To expose on the \
             LAN pass --bind 0.0.0.0; combine with --require-auth and \
             --auth-tokens-file for non-trusted networks."
        );
    }
    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    serve_with_header_timeout(listener, app).await
}

/// Serve `app` with a hyper connection-layer **header-read timeout** so a
/// slowloris client (one that opens a connection and dribbles request headers
/// forever) cannot pin an accept slot indefinitely.
///
/// `axum::serve` uses hyper's defaults, which impose NO timeout on the
/// header-read phase (the per-request scheduler `timeout_at` only engages
/// AFTER the request is fully parsed and admitted, so it does not protect this
/// phase). A blanket `tower_http::TimeoutLayer` is the wrong tool — it would
/// also abort legitimate long generations. So we drop to hyper's
/// `hyper_util::server::conn::auto::Builder` and set `header_read_timeout`
/// directly. `into_make_service_with_connect_info` is preserved (per-connection
/// `make_service.call(peer)`), so `ConnectInfo<SocketAddr>` — which
/// `rate_limit_middleware` reads — keeps working.
async fn serve_with_header_timeout(
    listener: tokio::net::TcpListener,
    app: Router,
) -> anyhow::Result<()> {
    use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
    use hyper_util::server::conn::auto::Builder;
    use tower::{Service, ServiceExt};

    /// Slow-header cutoff. Matches hyper's own historical default; long enough
    /// for any legitimate client to finish sending headers, short enough that a
    /// trickle connection is reaped quickly.
    const HEADER_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

    let mut make_service = app.into_make_service_with_connect_info::<std::net::SocketAddr>();

    loop {
        let (socket, peer_addr) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                // Transient accept errors (fd exhaustion, RST races) must not
                // kill the server — log and keep accepting.
                tracing::warn!("accept error: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                continue;
            }
        };

        // Build the per-connection tower service, wiring the peer address into
        // `ConnectInfo`. `IntoMakeServiceWithConnectInfo` is always ready and
        // infallible.
        let tower_service = match make_service.call(peer_addr).await {
            Ok(svc) => svc,
            Err(infallible) => match infallible {},
        };

        tokio::spawn(async move {
            let socket = TokioIo::new(socket);
            let hyper_service = hyper::service::service_fn(
                move |request: hyper::Request<hyper::body::Incoming>| {
                    tower_service.clone().oneshot(request)
                },
            );

            let mut builder = Builder::new(TokioExecutor::new());
            // A timer must be installed for the header-read timeout to fire.
            builder
                .http1()
                .timer(TokioTimer::new())
                .header_read_timeout(HEADER_READ_TIMEOUT);
            builder.http2().timer(TokioTimer::new());

            if let Err(err) = builder
                .serve_connection_with_upgrades(socket, hyper_service)
                .await
            {
                // Client-side disconnects / slow-header timeouts are expected
                // and noisy — keep them at debug.
                tracing::debug!("connection closed: {err}");
            }
        });
    }
}
