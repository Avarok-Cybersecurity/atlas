// SPDX-License-Identifier: AGPL-3.0-only

//! Rendering, tested as pure functions of a Score.

use super::super::compare::{RoundVerdict, TurnDelta};
use super::super::score::{RoundRecord, score};
use super::{metrics, summary, table};
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

/// A warm-cache round record; 992 is the smallest restore anchor the
/// 2026-08-12 run observed.
fn rec(round: usize, verdict: RoundVerdict) -> RoundRecord {
    RoundRecord {
        round,
        verdict,
        turn1_cached: Some(992),
    }
}

#[test]
fn metrics_carry_every_class_even_when_zero() {
    let replays: Vec<_> = (1..=3).map(|n| rec(n, RoundVerdict::Invariant)).collect();
    let m = metrics(&score(&replays));
    for key in [
        "rounds",
        "invariant",
        "jittered",
        "collapsed",
        "unmeasured",
        "min_cached_prompt_tokens",
    ] {
        assert!(m.contains_key(key), "missing metric {key}");
    }
    assert_eq!(m["rounds"], 3.0);
    assert_eq!(m["invariant"], 3.0);
    assert_eq!(m["collapsed"], 0.0);
    assert_eq!(m["min_cached_prompt_tokens"], 992.0);
}

#[test]
fn metrics_reflect_mixed_classes() {
    let replays = vec![
        rec(1, RoundVerdict::Invariant),
        rec(
            2,
            RoundVerdict::Jittered {
                turns: vec![delta(206)],
            },
        ),
        rec(
            3,
            RoundVerdict::Collapsed {
                turns: vec![delta(3)],
            },
        ),
        RoundRecord {
            round: 4,
            verdict: RoundVerdict::Unmeasured {
                reason: "reset".into(),
            },
            turn1_cached: None,
        },
    ];
    let m = metrics(&score(&replays));
    assert_eq!(m["rounds"], 4.0);
    assert_eq!(m["invariant"], 1.0);
    assert_eq!(m["jittered"], 1.0);
    assert_eq!(m["collapsed"], 1.0);
    assert_eq!(m["unmeasured"], 1.0);
}

#[test]
fn a_zero_cache_run_records_a_zero_metric() {
    // The BENCH.toml floor (min = 1.0) reads this key: a run that never
    // engaged the prefix cache must record 0 and fail at the record level,
    // where before the metric existed it recorded nothing and passed.
    let replays: Vec<_> = (1..=3)
        .map(|n| RoundRecord {
            round: n,
            verdict: RoundVerdict::Invariant,
            turn1_cached: Some(0),
        })
        .collect();
    let m = metrics(&score(&replays));
    assert_eq!(m["min_cached_prompt_tokens"], 0.0);
}

#[test]
fn collapsed_tile_is_red_only_when_present() {
    let clean: Vec<_> = (1..=3).map(|n| rec(n, RoundVerdict::Invariant)).collect();
    let s = summary(&score(&clean));
    assert_eq!(s[2].style, CellStyle::Good); // Collapsed

    let poisoned = vec![rec(
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
    let jittered = vec![rec(
        1,
        RoundVerdict::Jittered {
            turns: vec![delta(206)],
        },
    )];
    let s = summary(&score(&jittered));
    assert_eq!(s[1].style, CellStyle::Warn); // Jittered
}

#[test]
fn cache_tile_is_red_when_the_restore_path_never_ran() {
    let warm: Vec<_> = (1..=2).map(|n| rec(n, RoundVerdict::Invariant)).collect();
    let s = summary(&score(&warm));
    assert_eq!(s[4].label, "Min t1 cache");
    assert_eq!(s[4].style, CellStyle::Good);

    let cold = vec![RoundRecord {
        round: 1,
        verdict: RoundVerdict::Invariant,
        turn1_cached: Some(0),
    }];
    let s = summary(&score(&cold));
    assert_eq!(s[4].style, CellStyle::Bad);
}

#[test]
fn unmeasured_rows_are_attributed_to_their_actual_round() {
    // B5: the table used to hand "unmeasured" to the earliest round that was
    // neither jittered nor collapsed, so an unmeasured round 3 painted round
    // 1 as the transport failure and round 3 as invariant.
    let replays = vec![
        rec(1, RoundVerdict::Invariant),
        rec(2, RoundVerdict::Invariant),
        RoundRecord {
            round: 3,
            verdict: RoundVerdict::Unmeasured {
                reason: "connection reset".into(),
            },
            turn1_cached: None,
        },
    ];
    let t = table(&score(&replays));
    assert_eq!(t.rows[0][1].text, "invariant", "round 1 was measured");
    assert_eq!(t.rows[1][1].text, "invariant", "round 2 was measured");
    assert_eq!(t.rows[2][1].text, "unmeasured", "round 3 was the failure");
    assert!(
        t.rows[2][2].text.contains("connection reset"),
        "the recorded reason belongs in the row: {}",
        t.rows[2][2].text
    );
}
