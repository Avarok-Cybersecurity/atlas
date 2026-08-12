// SPDX-License-Identifier: AGPL-3.0-only

//! Rendering, tested as pure functions of a Score.

use super::super::compare::{RoundVerdict, TurnDelta};
use super::super::score::score;
use super::{metrics, summary};
use crate::result::CellStyle;

fn delta(replay_tokens: usize) -> TurnDelta {
    TurnDelta {
        turn: 2,
        ref_tokens: 200,
        replay_tokens,
        ref_finish: Some("stop".into()),
        replay_finish: Some("stop".into()),
    }
}

#[test]
fn metrics_carry_every_class_even_when_zero() {
    let replays: Vec<_> = (1..=3).map(|n| (n, RoundVerdict::Invariant)).collect();
    let m = metrics(&score(&replays));
    for key in ["rounds", "invariant", "jittered", "collapsed", "unmeasured"] {
        assert!(m.contains_key(key), "missing metric {key}");
    }
    assert_eq!(m["rounds"], 3.0);
    assert_eq!(m["invariant"], 3.0);
    assert_eq!(m["collapsed"], 0.0);
}

#[test]
fn metrics_reflect_mixed_classes() {
    let replays = vec![
        (1, RoundVerdict::Invariant),
        (
            2,
            RoundVerdict::Jittered {
                turns: vec![delta(206)],
            },
        ),
        (
            3,
            RoundVerdict::Collapsed {
                turns: vec![delta(3)],
            },
        ),
        (
            4,
            RoundVerdict::Unmeasured {
                reason: "reset".into(),
            },
        ),
    ];
    let m = metrics(&score(&replays));
    assert_eq!(m["rounds"], 4.0);
    assert_eq!(m["invariant"], 1.0);
    assert_eq!(m["jittered"], 1.0);
    assert_eq!(m["collapsed"], 1.0);
    assert_eq!(m["unmeasured"], 1.0);
}

#[test]
fn collapsed_tile_is_red_only_when_present() {
    let clean: Vec<_> = (1..=3).map(|n| (n, RoundVerdict::Invariant)).collect();
    let s = summary(&score(&clean));
    assert_eq!(s[2].style, CellStyle::Good); // Collapsed

    let poisoned = vec![(
        1,
        RoundVerdict::Collapsed {
            turns: vec![delta(3)],
        },
    )];
    let s = summary(&score(&poisoned));
    assert_eq!(s[2].style, CellStyle::Bad); // Collapsed
}

#[test]
fn jitter_tile_is_warn_only_when_present_never_red() {
    let jittered = vec![(
        1,
        RoundVerdict::Jittered {
            turns: vec![delta(206)],
        },
    )];
    let s = summary(&score(&jittered));
    assert_eq!(s[1].style, CellStyle::Warn); // Jittered
}
