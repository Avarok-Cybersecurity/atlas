// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn cfg(rpm: u64, tpm: u64) -> RateLimitConfig {
    RateLimitConfig {
        rpm,
        tpm,
        burst_rpm: rpm.max(1),
        burst_tpm: tpm.max(1),
    }
}

#[test]
fn disabled_limiter_always_allows() {
    let lim = RateLimiter::with_config(cfg(0, 0));
    for _ in 0..100 {
        let d = lim.admit("k", 10_000);
        assert!(d.allowed);
    }
}

#[test]
fn request_bucket_blocks_after_burst() {
    let lim = RateLimiter::with_config(cfg(2, 0));
    assert!(lim.admit("k", 0).allowed);
    assert!(lim.admit("k", 0).allowed);
    let d = lim.admit("k", 0);
    assert!(!d.allowed);
    assert!(matches!(d.denied_by, Some(DenialReason::Requests)));
    assert!(d.retry_after_secs >= 1);
}

#[test]
fn token_bucket_blocks_on_oversized_request() {
    let lim = RateLimiter::with_config(cfg(0, 1000));
    assert!(lim.admit("k", 600).allowed);
    // 600 consumed, 400 left; next 500 should deny.
    let d = lim.admit("k", 500);
    assert!(!d.allowed);
    assert!(matches!(d.denied_by, Some(DenialReason::Tokens)));
}

#[test]
fn keys_do_not_interfere() {
    let lim = RateLimiter::with_config(cfg(1, 0));
    assert!(lim.admit("a", 0).allowed);
    assert!(lim.admit("b", 0).allowed);
    assert!(!lim.admit("a", 0).allowed);
    assert!(!lim.admit("b", 0).allowed);
}

#[test]
fn identity_prefers_bearer_over_xff() {
    use axum::http::{HeaderMap, HeaderValue, header};
    let mut h = HeaderMap::new();
    h.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer sk-abc"),
    );
    h.insert("x-forwarded-for", HeaderValue::from_static("1.2.3.4"));
    let id = extract_identity(&h, None);
    assert!(id.starts_with("bearer:"));
}

#[test]
fn identity_falls_back_to_xff_then_peer() {
    use axum::http::{HeaderMap, HeaderValue};
    let mut h = HeaderMap::new();
    h.insert(
        "x-forwarded-for",
        HeaderValue::from_static("9.9.9.9, 1.1.1.1"),
    );
    assert_eq!(extract_identity(&h, None), "xff:9.9.9.9");
    let empty = HeaderMap::new();
    let peer: std::net::SocketAddr = "127.0.0.1:9000".parse().unwrap();
    assert_eq!(extract_identity(&empty, Some(peer)), "peer:127.0.0.1");
}

#[test]
fn token_refund_restores_budget() {
    let lim = RateLimiter::with_config(cfg(0, 1000));
    assert!(lim.admit("k", 900).allowed);
    lim.refund_tokens("k", 500);
    assert!(lim.admit("k", 500).allowed);
}
