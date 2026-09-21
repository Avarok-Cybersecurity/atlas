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
        prompt_tokens: (distinct == 1).then_some(100),
        wall_ms: 1,
        errors: (0..errs).map(|i| format!("e{i}")).collect(),
    }
}

#[test]
fn a_clean_level_is_ok() {
    assert!(r(4, 4, 4, 1, 0).ok());
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
fn the_levels_cross_the_single_stream_boundary() {
    assert!(LEVELS.contains(&1), "a baseline to compare against");
    assert!(
        LEVELS.iter().any(|&c| c > 1),
        "a single-stream-only sweep would exercise none of the shared state"
    );
    assert!(LEVELS.windows(2).all(|w| w[0] < w[1]), "ascending");
}

#[test]
fn every_level_must_preserve_the_single_stream_geometry() {
    let baseline = r(1, 1, 1, 1, 0);
    let mut changed = r(2, 2, 2, 1, 0);
    changed.prompt_tokens = Some(200);
    let high = r(4, 4, 4, 1, 0);

    assert!(changed.ok(), "the C=2 replies agree with each other");
    assert_eq!(
        changed.geometry_detail(baseline.prompt_tokens),
        "200 prompt tokens, C=1 baseline 100"
    );
    assert!(
        !sweep_ok(&[baseline, changed, high]),
        "internal agreement is not enough when C=2 changed from the C=1 geometry"
    );
}

#[test]
fn an_incomplete_concurrency_sweep_is_not_clean() {
    assert!(
        !sweep_ok(&[r(1, 1, 1, 1, 0), r(2, 2, 2, 1, 0)]),
        "omitting the highest configured level leaves the shared-state claim untested"
    );
}
