// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the concurrency leg's scoring. The fan-out itself needs a served
//! model; what is checkable here is that a result is judged correctly.

use super::*;

fn r(conc: usize, returned: usize, correct: usize, distinct: usize, errs: usize) -> LevelResult {
    LevelResult {
        conc,
        returned,
        correct,
        distinct_token_counts: distinct,
        wall_ms: 1,
        errors: (0..errs).map(|i| format!("e{i}")).collect(),
    }
}

#[test]
fn a_clean_level_is_ok() {
    assert!(r(4, 4, 4, 1, 0).ok());
}

#[test]
fn a_dropped_reply_is_not_ok() {
    assert!(!r(4, 3, 3, 1, 0).ok());
}

/// ★ The case this leg exists for. Every request returned, none errored, and
/// the geometry agreed — but a reply was WRONG. That is what cross-request
/// contamination looks like: request A answering with request B's content.
/// A survival-only check ("did it 200?") passes this and should not.
#[test]
fn every_reply_returning_is_not_enough_if_one_is_wrong() {
    let bad = r(4, 4, 3, 1, 0);
    assert_eq!(bad.returned, bad.conc, "nothing was dropped");
    assert!(bad.errors.is_empty(), "nothing errored");
    assert!(!bad.ok(), "but one reply was wrong, so the level must fail");
}

/// Identical requests must produce identical prompt-token counts. Two
/// different counts from the same body means the shared vision buffers were
/// indexed per-request incorrectly.
#[test]
fn disagreeing_geometry_across_identical_requests_is_not_ok() {
    assert!(!r(4, 4, 4, 2, 0).ok());
}

#[test]
fn an_error_is_not_ok_even_if_the_rest_were_fine() {
    assert!(!r(4, 4, 4, 1, 1).ok());
}

#[test]
fn the_levels_cross_the_single_stream_boundary() {
    assert!(LEVELS.contains(&1), "a baseline to compare against");
    assert!(
        LEVELS.iter().any(|&c| c > 1),
        "a single-stream-only sweep would exercise none of the shared state"
    );
    assert!(LEVELS.windows(2).all(|w| w[0] < w[1]), "ascending");
}
