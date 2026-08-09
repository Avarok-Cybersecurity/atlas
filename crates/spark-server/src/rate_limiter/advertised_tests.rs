// SPDX-License-Identifier: AGPL-3.0-only

//! What the limiter ADVERTISES, as distinct from what it enforces.
//!
//! Split from `rate_limiter.rs` only to stay under the repository's
//! per-file cap; it is one concern and moves as a unit.

use super::*;

#[test]
fn an_unenforced_axis_is_not_advertised_as_a_limit_of_one() {
    // burst_tpm floors at 1 so the bucket maths never divides by zero, but
    // with tpm == 0 the token axis is not enforced. Advertising 1 told a
    // client honouring these headers it had a single token left.
    let cfg = RateLimitConfig {
        rpm: 3,
        tpm: 0,
        burst_rpm: 3,
        burst_tpm: 1,
    };
    let d = RateLimiter::with_config(cfg).admit("k", 8192);
    assert_eq!(
        d.requests.limit, 3,
        "the enforced axis reports its real limit"
    );
    assert!(
        d.tokens.limit > 1_000_000,
        "the unenforced axis reports effectively unlimited, not 1: {}",
        d.tokens.limit
    );
    // A limit of "unlimited" beside a remaining of 1 is a contradiction the
    // client has to resolve, and it will resolve it the cautious way.
    assert!(
        d.tokens.remaining > 1_000_000,
        "remaining must agree with the advertised limit: {}",
        d.tokens.remaining
    );
}

#[test]
fn both_axes_report_their_real_limits_when_both_are_enforced() {
    let cfg = RateLimitConfig {
        rpm: 5,
        tpm: 900,
        burst_rpm: 5,
        burst_tpm: 900,
    };
    let d = RateLimiter::with_config(cfg).admit("k", 1);
    assert_eq!(d.requests.limit, 5);
    assert_eq!(d.tokens.limit, 900);
}
