// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the batched-verify staging helpers (`verify_e2.rs`).
//!
//! [`value_switch_armed`] must keep the house VALUE convention for the three
//! verify switches hoisted out of the per-step hot path. An inverted or
//! loosened predicate would arm `ATLAS_K4_DIAG` (which forces the verify path
//! EAGER, destroying the graph replay) for anyone who exports it as `=0`.

use super::value_switch_armed;

/// The hoisted verify switches are VALUE switches: only the literal `"1"`
/// arms them. Pins the exact predicate the three per-step `std::env::var`
/// reads used to spell inline, so hoisting them behind a `OnceLock` cannot
/// have widened (`presence`) or inverted the semantics.
#[test]
fn value_switch_is_armed_only_by_a_literal_one() {
    assert!(value_switch_armed(Some("1")));
    for raw in [
        None,
        Some(""),
        Some("0"),
        Some("2"),
        Some("true"),
        Some(" 1"),
    ] {
        assert!(
            !value_switch_armed(raw),
            "{raw:?} must not arm a VALUE switch"
        );
    }
}
