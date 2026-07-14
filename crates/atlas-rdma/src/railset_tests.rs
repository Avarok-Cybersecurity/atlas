// SPDX-License-Identifier: AGPL-3.0-only
//
// Guard tests for `RailSet::complete`'s rail/param pairing. `complete` itself
// needs a live NIC (`Verbs::create`), so the check is factored into the pure
// `check_rail_count` and exercised here without hardware.

use super::check_rail_count;

#[test]
fn equal_counts_are_ok() {
    assert!(check_rail_count(2, 2, "peer").is_ok());
    assert!(check_rail_count(0, 0, "peer").is_ok());
}

#[test]
fn short_server_slice_is_refused_not_truncated() {
    // The regression this guards: a bare `zip` would connect 1 of 2 rails and
    // return Ok, leaving rail 1 stuck in INIT. The `peer` arg is just the label
    // the error interpolates (real endpoints are env-resolved upstream); a
    // distinctive fixture value lets us assert it is echoed.
    let e = check_rail_count(1, 2, "test-peer").unwrap_err().to_string();
    assert!(e.contains("test-peer"), "peer named: {e}");
    assert!(
        e.contains("1 rail params for 2 client rails"),
        "counts reported: {e}"
    );
}

#[test]
fn long_server_slice_is_also_refused() {
    assert!(check_rail_count(3, 2, "peer").is_err());
}
