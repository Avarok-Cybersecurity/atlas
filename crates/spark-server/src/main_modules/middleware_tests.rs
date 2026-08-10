// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the response-header middleware.

#[test]
fn the_compat_stubs_do_not_overwrite_real_rate_limit_headers() {
    // This layer runs OUTSIDE rate_limit_middleware, so `insert` overwrote the
    // limiter's real numbers with "unlimited, nothing used, no reset" on every
    // response. A client honouring these headers would never back off and would
    // drive straight into the 429s the limiter exists to prevent.
    use axum::http::{HeaderMap, HeaderName, HeaderValue};
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("x-ratelimit-limit-requests"),
        HeaderValue::from_static("3"),
    );
    super::apply_compat_stubs(&mut headers);

    assert_eq!(
        headers
            .get("x-ratelimit-limit-requests")
            .and_then(|v| v.to_str().ok()),
        Some("3"),
        "the limiter's real value survives"
    );
    assert_eq!(
        headers
            .get("x-ratelimit-limit-tokens")
            .and_then(|v| v.to_str().ok()),
        Some("1000000000"),
        "a field the limiter did not set still gets its stub"
    );
}

// ---------------------------------------------------------------------------
// #429: refuse inference once the GPU context is dead
// ---------------------------------------------------------------------------

const GPU_FAULT: &str = "the CUDA context is destroyed";

/// POSITIVE: an inference request on a faulted server is refused with 503 and
/// an OpenAI-shaped error body, not admitted and 500'd deep in the driver.
///
/// PROVEN BY: making `fault_rejection` return `None` unconditionally turns
/// this red on the `is_some` assertion.
#[test]
fn inference_is_refused_once_the_gpu_has_faulted() {
    let got = super::fault_rejection("/v1/chat/completions", Some(GPU_FAULT));
    let (code, body) = got.expect("a faulted server must refuse inference");
    assert_eq!(code, axum::http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "gpu_fault");
    assert_eq!(body["error"]["type"], "server_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains(GPU_FAULT)),
        "the reason must reach the client: {body}"
    );
}

/// NEGATIVE: a healthy server admits the same request untouched. Without this
/// the middleware would be a total outage rather than a fault handler.
///
/// PROVEN BY: making `fault_rejection` return `Some(..)` unconditionally turns
/// this red.
#[test]
fn a_healthy_server_admits_inference() {
    assert!(super::fault_rejection("/v1/chat/completions", None).is_none());
}

/// NEGATIVE, and the one that keeps the fault diagnosable: `/health` must
/// still answer while faulted. It is how an operator and an orchestrator learn
/// what happened; refusing it would convert a reported fault into a silent one.
///
/// PROVEN BY: deleting the `if !path.starts_with("/v1/")` early return turns
/// this red — the middleware then eats its own health endpoints.
#[test]
fn health_endpoints_still_answer_while_faulted() {
    for path in ["/health", "/health/live", "/metrics", "/hardware"] {
        assert!(
            super::fault_rejection(path, Some(GPU_FAULT)).is_none(),
            "{path} must remain reachable during a fault"
        );
    }
}
