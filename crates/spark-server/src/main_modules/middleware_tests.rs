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
