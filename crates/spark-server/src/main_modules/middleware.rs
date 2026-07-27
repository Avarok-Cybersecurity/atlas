// SPDX-License-Identifier: AGPL-3.0-only

//! Axum middleware: observability, auth, rate-limiting.

use std::sync::Arc;

use crate::main_modules::AppState;
use crate::rate_limiter;

/// OpenAI-compatible observability headers. Injects on every `/v1/*`
/// response:
/// - `x-request-id`: UUID v4 generated per-request (re-used if client
///   supplied one, letting callers correlate logs end-to-end).
/// - `openai-processing-ms`: server wall-clock time in milliseconds.
/// - `x-ratelimit-*`: static "unlimited" stubs so clients that
///   parse them for backoff don't treat missing headers as unlimited
///   (some SDKs assume 0 = exhausted).
/// - `openai-organization`, `openai-version`: static stubs for
///   parity with `api.openai.com` — several wrappers log these.
pub(crate) async fn openai_observability_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{HeaderName, HeaderValue};
    let start = std::time::Instant::now();
    let is_v1 = req.uri().path().starts_with("/v1/");
    let incoming_req_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let mut resp = next.run(req).await;
    if !is_v1 {
        return resp;
    }
    let headers = resp.headers_mut();
    let rid = incoming_req_id.unwrap_or_else(|| format!("req_{}", crate::ids::uuid_v4()));
    if let Ok(v) = HeaderValue::from_str(&rid) {
        headers.insert(HeaderName::from_static("x-request-id"), v);
    }
    let elapsed_ms = start.elapsed().as_millis();
    if let Ok(v) = HeaderValue::from_str(&elapsed_ms.to_string()) {
        headers.insert(HeaderName::from_static("openai-processing-ms"), v);
    }
    // Static "effectively unlimited" stubs (Atlas does not enforce rate limits).
    for (k, v) in [
        ("x-ratelimit-limit-requests", "1000000"),
        ("x-ratelimit-remaining-requests", "999999"),
        ("x-ratelimit-reset-requests", "0s"),
        ("x-ratelimit-limit-tokens", "1000000000"),
        ("x-ratelimit-remaining-tokens", "999999999"),
        ("x-ratelimit-reset-tokens", "0s"),
        ("openai-organization", "atlas-local"),
        ("openai-version", "2026-01-01"),
    ] {
        if let Ok(val) = HeaderValue::from_str(v) {
            headers.insert(HeaderName::from_static(k), val);
        }
    }
    resp
}

/// Bearer-token gate. Active when the operator passed `--require-auth`
/// (which lands in `AppState.auth` as `Some(...)`); otherwise this is a
/// pass-through. When active, requests to `/v1/*`, `/tokenize`, and
/// `/detokenize` must carry `Authorization: Bearer <token>` matching one
/// of the loaded tokens. Health / metrics / liveness paths stay open
/// (they're scrape targets / discovery endpoints, expected to be
/// firewall-protected at the network layer).
///
/// Token comparison is constant-time per candidate (see `auth::AuthConfig`)
/// so an attacker can't recover the secret by measuring early-exit
/// timings. The error body matches the OpenAI shape so client SDKs
/// surface "missing_api_key" / "invalid_api_key" the same way they do
/// against `api.openai.com`.
pub(crate) async fn require_auth_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let Some(auth_cfg) = state.auth.as_ref() else {
        return next.run(req).await;
    };
    let path = req.uri().path();
    let needs_auth = path.starts_with("/v1/") || path == "/tokenize" || path == "/detokenize";
    if !needs_auth {
        return next.run(req).await;
    }
    let presented_token = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(str::trim);
    let (status, code, message) = match presented_token {
        None => (
            axum::http::StatusCode::UNAUTHORIZED,
            "missing_api_key",
            "Missing Authorization: Bearer header",
        ),
        Some(t) if !auth_cfg.validate(t.as_bytes()) => (
            axum::http::StatusCode::UNAUTHORIZED,
            "invalid_api_key",
            "Invalid bearer token",
        ),
        Some(_) => return next.run(req).await,
    };
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": null,
            "code": code,
        }
    });
    (status, axum::Json(body)).into_response()
}

/// Per-identity rate-limit middleware. When the limiter is enabled via
/// `ATLAS_RATE_LIMIT_RPM` / `ATLAS_RATE_LIMIT_TPM`, admission is checked
/// before dispatching to the handler; denied requests short-circuit with
/// 429 + OpenAI error body + `retry-after` header.
///
/// Identity precedence (from `rate_limiter::extract_identity`):
/// Bearer token → X-Forwarded-For → socket peer addr.
///
/// Headers emitted in both allow and deny paths:
///   x-ratelimit-limit-{requests,tokens}
///   x-ratelimit-remaining-{requests,tokens}
///   x-ratelimit-reset-{requests,tokens}
///
/// These overwrite the static "unlimited" stubs from
/// `openai_observability_middleware`. When the limiter is disabled, this
/// middleware is a pass-through (the observability stubs stand).
pub(crate) async fn rate_limit_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{HeaderName, HeaderValue, StatusCode};
    use axum::response::IntoResponse;

    // Only apply to /v1/* routes — health/metrics/tokenize stay open.
    let is_v1 = req.uri().path().starts_with("/v1/");
    if !is_v1 || !state.rate_limiter.config().is_enabled() {
        return next.run(req).await;
    }

    // Peer addr comes from the ConnectInfo extension injected by
    // `into_make_service_with_connect_info`. Falls through to None when
    // running under a unit-test harness that doesn't set it.
    let peer = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0);
    let identity = rate_limiter::extract_identity(req.headers(), peer);

    // Estimated token cost = this request's conservative upper bound.
    // We use max_seq_len as the ceiling; true-up is applied on completion
    // for streaming paths (handlers call `refund_tokens` with the actual
    // usage). This over-counts for small requests but prevents a single
    // client from consuming the whole TPM budget in one burst.
    let estimated = state.max_seq_len as u64;

    let decision = state.rate_limiter.admit(&identity, estimated);
    if !decision.allowed {
        let (param, code) = match decision.denied_by {
            Some(rate_limiter::DenialReason::Requests) => ("requests", "rate_limit_exceeded"),
            Some(rate_limiter::DenialReason::Tokens) => ("tokens", "rate_limit_exceeded"),
            None => ("", "rate_limit_exceeded"),
        };
        let body = serde_json::json!({
            "error": {
                "message": format!("Rate limit exceeded for {param}. Retry after {}s.", decision.retry_after_secs),
                "type": "rate_limit_exceeded",
                "param": param,
                "code": code,
            }
        });
        let mut resp = (StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response();
        apply_rate_headers(resp.headers_mut(), &decision);
        if let Ok(v) = HeaderValue::from_str(&decision.retry_after_secs.to_string()) {
            resp.headers_mut()
                .insert(HeaderName::from_static("retry-after"), v);
        }
        return resp;
    }

    // Stash the identity + reservation on the request so handlers can
    // refund the over-estimated portion once the true usage is known.
    let mut req = req;
    req.extensions_mut().insert(rate_limiter::RequestContext {
        identity: identity.clone(),
        reserved_tokens: estimated,
    });
    let mut resp = next.run(req).await;
    apply_rate_headers(resp.headers_mut(), &decision);
    resp
}

pub(crate) fn apply_rate_headers(
    headers: &mut axum::http::HeaderMap,
    d: &rate_limiter::RateDecision,
) {
    use axum::http::{HeaderName, HeaderValue};
    let set = |h: &mut axum::http::HeaderMap, k: &'static str, v: String| {
        if let Ok(val) = HeaderValue::from_str(&v) {
            h.insert(HeaderName::from_static(k), val);
        }
    };
    set(
        headers,
        "x-ratelimit-limit-requests",
        d.requests.limit.to_string(),
    );
    set(
        headers,
        "x-ratelimit-remaining-requests",
        d.requests.remaining.to_string(),
    );
    set(
        headers,
        "x-ratelimit-reset-requests",
        format!("{}s", d.requests.reset_secs),
    );
    set(
        headers,
        "x-ratelimit-limit-tokens",
        d.tokens.limit.to_string(),
    );
    set(
        headers,
        "x-ratelimit-remaining-tokens",
        d.tokens.remaining.to_string(),
    );
    set(
        headers,
        "x-ratelimit-reset-tokens",
        format!("{}s", d.tokens.reset_secs),
    );
}
