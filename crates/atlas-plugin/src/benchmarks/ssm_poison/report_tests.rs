// SPDX-License-Identifier: AGPL-3.0-only

//! Rendering, tested as pure functions of a Score.

use super::super::compare::RoundVerdict;
use super::super::score::score;
use super::{metrics, summary};
use crate::result::CellStyle;

fn score_from(replays: Vec<(usize, RoundVerdict)>) -> super::super::score::Score {
    score(&replays)
}

#[test]
fn metrics_carry_every_class_even_when_zero() {
    let replays: Vec<_> = (1..=3).map(|n| (n, RoundVerdict::Invariant)).collect();
    let s = score_from(replays);
    let m = metrics(&s);
    for key in ["rounds", "invariant", "diverged", "unmeasured"] {
        assert!(m.contains_key(key), "missing metric {key}");
    }
    assert_eq!(m["rounds"], 3.0);
    assert_eq!(m["invariant"], 3.0);
    assert_eq!(m["diverged"], 0.0);
    assert_eq!(m["unmeasured"], 0.0);
}

#[test]
fn metrics_reflect_a_divergence() {
    let replays = vec![
        (1, RoundVerdict::Invariant),
        (2, RoundVerdict::Diverged { turns: vec![3] }),
        (
            3,
            RoundVerdict::Unmeasured {
                reason: "reset".into(),
            },
        ),
    ];
    let m = metrics(&score_from(replays));
    assert_eq!(m["rounds"], 3.0);
    assert_eq!(m["invariant"], 1.0);
    assert_eq!(m["diverged"], 1.0);
    assert_eq!(m["unmeasured"], 1.0);
}

#[test]
fn summary_headline_is_green_only_when_all_invariant() {
    let clean: Vec<_> = (1..=4).map(|n| (n, RoundVerdict::Invariant)).collect();
    let s = summary(&score_from(clean));
    assert_eq!(s[0].style, CellStyle::Good);

    let dirty = vec![
        (1, RoundVerdict::Invariant),
        (2, RoundVerdict::Diverged { turns: vec![1] }),
    ];
    let s = summary(&score_from(dirty));
    assert_eq!(s[0].style, CellStyle::Bad);
    assert_eq!(s[1].style, CellStyle::Bad);
}
