// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (strict `ATLAS_*` parsing).
//!
//! Every case here is driven through the PURE entry points — `parse_min` and
//! `RateLimitConfig::from_raw` — never `set_var`. The environment is
//! process-global, so a test that writes it races every other test in the
//! binary and passes or fails by scheduling.

use super::parse_min;

/// The defect, at the level the operator experiences it: a typo in the rate
/// limit used to leave the server UNLIMITED, quietly.
///
/// Asserted against `RateLimitConfig` rather than `parse_min` because that is
/// where the damage was — `.parse().ok().unwrap_or(0)` and `0` means "off".
#[test]
fn a_typo_in_the_rate_limit_is_refused_instead_of_disabling_the_limit() {
    let err = crate::rate_limiter::RateLimitConfig::from_raw(Some("1oo"), None, None, None)
        .expect_err("a malformed rate limit must not be accepted");
    assert!(
        err.contains("ATLAS_RATE_LIMIT_RPM"),
        "the message must name the key the operator has to fix: {err}"
    );
    assert!(
        err.contains("1oo"),
        "the message must quote the value it rejected: {err}"
    );
    assert!(
        err.contains("fix:"),
        "a diagnostic without a fix is half of one: {err}"
    );
}

/// The whole point of refusing: the old code turned this exact input into a
/// server with no rate limiting at all.
#[test]
fn the_old_silent_fallback_would_have_produced_an_unlimited_server() {
    // What the previous implementation did, spelled out so the regression is
    // legible: parse, discard the error, take the default.
    let silently: u64 = "1oo".parse().ok().unwrap_or(0);
    assert_eq!(silently, 0, "0 is the value that means NO rate limit");
    // What it does now.
    assert!(crate::rate_limiter::RateLimitConfig::from_raw(Some("1oo"), None, None, None).is_err());
}

#[test]
fn a_valid_rate_limit_still_parses_and_burst_still_defaults_to_the_rate() {
    let cfg = crate::rate_limiter::RateLimitConfig::from_raw(Some("100"), Some("5000"), None, None)
        .expect("well-formed values must be accepted");
    assert_eq!(cfg.rpm, 100);
    assert_eq!(cfg.tpm, 5000);
    // Unset burst must stay "defaults to the sustained rate" — the reason the
    // parse returns Option rather than folding the default in early.
    assert_eq!(cfg.burst_rpm, 100);
    assert_eq!(cfg.burst_tpm, 5000);
}

#[test]
fn every_rate_limit_key_is_checked_not_just_the_first() {
    for (i, name) in [
        "ATLAS_RATE_LIMIT_RPM",
        "ATLAS_RATE_LIMIT_TPM",
        "ATLAS_RATE_LIMIT_BURST_RPM",
        "ATLAS_RATE_LIMIT_BURST_TPM",
    ]
    .iter()
    .enumerate()
    {
        let mut raw: [Option<&str>; 4] = [None; 4];
        raw[i] = Some("nope");
        let err = crate::rate_limiter::RateLimitConfig::from_raw(raw[0], raw[1], raw[2], raw[3])
            .expect_err(&format!("{name} must be validated"));
        assert!(err.contains(name), "wrong key named for {name}: {err}");
    }
}

#[test]
fn unset_and_blank_mean_unset_not_an_error() {
    assert_eq!(parse_min::<u64>("K", None, 0, "m"), Ok(None));
    assert_eq!(parse_min::<u64>("K", Some(""), 0, "m"), Ok(None));
    assert_eq!(parse_min::<u64>("K", Some("   "), 0, "m"), Ok(None));
}

#[test]
fn surrounding_whitespace_is_tolerated() {
    assert_eq!(parse_min::<u64>("K", Some(" 42 "), 0, "m"), Ok(Some(42)));
}

/// The bound has to be enforced, not just the syntax: `ATLAS_STORE_MAX_ENTRIES=0`
/// used to be filtered back to the 10 000 default, so an operator asking for a
/// disabled store silently got a full one.
#[test]
fn a_value_below_the_minimum_is_refused_and_the_minimum_is_named() {
    let err = parse_min::<usize>("ATLAS_STORE_MAX_ENTRIES", Some("0"), 1, "entries kept")
        .expect_err("0 is below the stated minimum of 1");
    assert!(err.contains("ATLAS_STORE_MAX_ENTRIES"), "{err}");
    assert!(err.contains(">= 1"), "must name the bound: {err}");
    assert!(err.contains("fix:"), "{err}");
}

/// A negative value must not be read as "unset" or wrap into a huge unsigned.
#[test]
fn a_negative_value_is_refused_for_an_unsigned_setting() {
    assert!(parse_min::<u64>("ATLAS_STORE_TTL_SECONDS", Some("-1"), 1, "seconds").is_err());
}

/// The unit suffix an operator actually types. `1h` used to become the 86 400
/// default — a 24-hour TTL for someone who asked for one hour.
#[test]
fn a_duration_with_a_unit_suffix_is_refused_rather_than_read_as_the_default() {
    let err = parse_min::<u64>("ATLAS_STORE_TTL_SECONDS", Some("1h"), 1, "seconds")
        .expect_err("`1h` is not a number of seconds");
    assert!(err.contains("1h"), "{err}");
    assert!(
        err.contains("whole number"),
        "must say what form is expected: {err}"
    );
}

/// The message has to survive being the only thing the operator sees, so it
/// must not depend on a `why:`-less rendering somewhere else.
#[test]
fn the_message_carries_what_why_and_fix() {
    let err = parse_min::<u64>("ATLAS_X", Some("bad"), 0, "what X controls").unwrap_err();
    assert!(err.contains("ATLAS_X=\"bad\""), "{err}");
    assert!(err.contains("why: what X controls"), "{err}");
    assert!(err.contains("fix:"), "{err}");
}
