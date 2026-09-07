// SPDX-License-Identifier: AGPL-3.0-only

//! Rendering, tested as pure functions of a Score.

use super::super::compare::{RoundVerdict, TurnDelta};
use super::super::score::{RoundRecord, score};
use super::{metrics, summary, table};
use crate::result::{CellStyle, Stat};

fn stat_projection(stats: &[Stat]) -> Vec<(&str, &str, &str, CellStyle)> {
    stats
        .iter()
        .map(|stat| {
            (
                stat.label.as_str(),
                stat.value.as_str(),
                stat.unit.as_str(),
                stat.style,
            )
        })
        .collect()
}

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
    assert_eq!(
        m,
        std::collections::BTreeMap::from([
            ("collapsed".into(), 0.0),
            ("invariant".into(), 3.0),
            ("jittered".into(), 0.0),
            ("min_cached_prompt_tokens".into(), 992.0),
            ("rounds".into(), 3.0),
            ("unmeasured".into(), 0.0),
        ])
    );
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
    assert_eq!(
        m,
        std::collections::BTreeMap::from([
            ("collapsed".into(), 1.0),
            ("invariant".into(), 1.0),
            ("jittered".into(), 1.0),
            ("min_cached_prompt_tokens".into(), 992.0),
            ("rounds".into(), 4.0),
            ("unmeasured".into(), 1.0),
        ])
    );
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
    assert_eq!(
        stat_projection(&s),
        [
            ("Invariant", "3/3", "", CellStyle::Good),
            ("Jittered", "0", "", CellStyle::Good),
            ("Collapsed", "0", "", CellStyle::Good),
            ("Unmeasured", "0", "", CellStyle::Good),
            ("Min t1 cache", "992", "tok", CellStyle::Good),
        ]
    );

    let poisoned = vec![rec(
        1,
        RoundVerdict::Collapsed {
            turns: vec![delta(3)],
        },
    )];
    let s = summary(&score(&poisoned));
    assert_eq!(
        stat_projection(&s),
        [
            ("Invariant", "0/1", "", CellStyle::Neutral),
            ("Jittered", "0", "", CellStyle::Good),
            ("Collapsed", "1", "", CellStyle::Bad),
            ("Unmeasured", "0", "", CellStyle::Good),
            ("Min t1 cache", "992", "tok", CellStyle::Good),
        ]
    );
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
    assert_eq!(
        stat_projection(&s),
        [
            ("Invariant", "0/1", "", CellStyle::Neutral),
            ("Jittered", "1", "", CellStyle::Warn),
            ("Collapsed", "0", "", CellStyle::Good),
            ("Unmeasured", "0", "", CellStyle::Good),
            ("Min t1 cache", "992", "tok", CellStyle::Good),
        ]
    );
}

#[test]
fn cache_tile_is_red_when_the_restore_path_never_ran() {
    let warm: Vec<_> = (1..=2).map(|n| rec(n, RoundVerdict::Invariant)).collect();
    let s = summary(&score(&warm));
    assert_eq!(
        stat_projection(&s),
        [
            ("Invariant", "2/2", "", CellStyle::Good),
            ("Jittered", "0", "", CellStyle::Good),
            ("Collapsed", "0", "", CellStyle::Good),
            ("Unmeasured", "0", "", CellStyle::Good),
            ("Min t1 cache", "992", "tok", CellStyle::Good),
        ]
    );

    let cold = vec![RoundRecord {
        round: 1,
        verdict: RoundVerdict::Invariant,
        turn1_cached: Some(0),
    }];
    let s = summary(&score(&cold));
    assert_eq!(
        stat_projection(&s),
        [
            ("Invariant", "1/1", "", CellStyle::Good),
            ("Jittered", "0", "", CellStyle::Good),
            ("Collapsed", "0", "", CellStyle::Good),
            ("Unmeasured", "0", "", CellStyle::Good),
            ("Min t1 cache", "0", "tok", CellStyle::Bad),
        ]
    );
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
    assert_eq!(
        t.rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| (cell.text.as_str(), cell.style))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        [
            vec![
                ("r1", CellStyle::Neutral),
                ("invariant", CellStyle::Good),
                ("", CellStyle::Neutral),
            ],
            vec![
                ("r2", CellStyle::Neutral),
                ("invariant", CellStyle::Good),
                ("", CellStyle::Neutral),
            ],
            vec![
                ("r3", CellStyle::Neutral),
                ("unmeasured", CellStyle::Bad),
                (
                    "invariant not proven this round — connection reset",
                    CellStyle::Neutral,
                ),
            ],
        ]
    );
    let stats = summary(&score(&replays));
    let unmeasured = stats
        .iter()
        .find(|stat| stat.label == "Unmeasured")
        .expect("unmeasured tile");
    assert_eq!(unmeasured.style, CellStyle::Bad);
}

/// ★ "The cache gave back nothing" and "no round reported a figure" are
/// different findings and must not share a value.
///
/// `min_turn1_cached` is `None` when not one replay produced a turn-1
/// attestation — every replay errored, or came back with no turns. Writing 0
/// for that stamped the record with the VACUITY signature the metric exists to
/// detect, so the gate's failure line read "min_cached_prompt_tokens 0 < 1" —
/// "the prefix cache restored nothing" — for a run in which nothing was ever
/// measured. Both still fail the floor; only one of them is true.
#[test]
fn an_unmeasured_cache_is_absent_from_the_record_not_zero() {
    let replays: Vec<_> = (1..=3)
        .map(|n| RoundRecord {
            round: n,
            verdict: RoundVerdict::Unmeasured {
                reason: "connection reset".into(),
            },
            turn1_cached: None,
        })
        .collect();
    let m = metrics(&score(&replays));
    assert!(
        !m.contains_key("min_cached_prompt_tokens"),
        "an unmeasured cache must not be written as the vacuity value: {m:?}"
    );
    // The rest of the record is unaffected — the run still says, loudly, that
    // three rounds went unmeasured.
    assert_eq!(m["rounds"], 3.0);
    assert_eq!(m["unmeasured"], 3.0);
    // And a REAL zero is still recorded as a zero, so the key still
    // discriminates. (`a_zero_cache_run_records_a_zero_metric` above pins
    // that direction; asserted here too so one edit cannot silence both.)
    let vacuous: Vec<_> = (1..=3)
        .map(|n| RoundRecord {
            round: n,
            verdict: RoundVerdict::Invariant,
            turn1_cached: Some(0),
        })
        .collect();
    assert_eq!(metrics(&score(&vacuous))["min_cached_prompt_tokens"], 0.0);
}
