// SPDX-License-Identifier: AGPL-3.0-only

//! How a poisoning run is presented. Pure functions of the [`super::score::Score`],
//! the same split `contamination/report.rs` uses: run logic and rendering read
//! separately, and everything here is table-testable with no server.

use std::collections::BTreeMap;

use super::compare::TurnDelta;
use super::score::Score;
use crate::result::{Cell, CellStyle, Column, ResultTable, Stat};

fn delta_detail(turns: &[TurnDelta]) -> String {
    turns
        .iter()
        .map(|t| {
            format!(
                "t{}: {}->{}tok fin {:?}->{:?}",
                t.turn, t.ref_tokens, t.replay_tokens, t.ref_finish, t.replay_finish
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// One row per replay round, in round order.
pub(super) fn table(s: &Score) -> ResultTable {
    let mut t = ResultTable::new(
        "REPLAY ROUNDS",
        vec![
            Column::left("Round", 6),
            Column::left("Result", 14),
            Column::left("Detail", 56),
        ],
    );
    let jitter_map: BTreeMap<usize, &Vec<TurnDelta>> =
        s.jittered_rounds.iter().map(|(n, d)| (*n, d)).collect();
    let collapse_map: BTreeMap<usize, &Vec<TurnDelta>> =
        s.collapsed_rounds.iter().map(|(n, d)| (*n, d)).collect();
    // Unmeasured rows come from the Score's own per-round records. The old
    // rendering counted unmeasured rounds and painted the label onto the
    // EARLIEST rows that had no other label, so an unmeasured round 9 could
    // be reported as round 1 — sending whoever reads the record to the wrong
    // place in the serve log.
    let unmeasured_map: BTreeMap<usize, &str> = s
        .unmeasured_rounds
        .iter()
        .map(|(n, r)| (*n, r.as_str()))
        .collect();
    for round in 1..=s.rounds {
        let (what, style, detail) = if let Some(turns) = collapse_map.get(&round) {
            ("COLLAPSED".to_string(), CellStyle::Bad, delta_detail(turns))
        } else if let Some(turns) = jitter_map.get(&round) {
            ("jittered".to_string(), CellStyle::Warn, delta_detail(turns))
        } else if let Some(reason) = unmeasured_map.get(&round) {
            (
                "unmeasured".into(),
                CellStyle::Bad,
                format!("invariant not proven this round — {reason}"),
            )
        } else {
            ("invariant".into(), CellStyle::Good, String::new())
        };
        t.push(vec![
            Cell::new(format!("r{round}")),
            Cell::styled(what, style),
            Cell::new(detail),
        ]);
    }
    t
}

/// The headline tiles. `Collapsed` is the gate's real bar — Good only at 0.
/// `Jittered` is informational (Warn only when present): restore jitter is a
/// healthy engine property, and painting it red would make the tile lie.
pub(super) fn summary(s: &Score) -> Vec<Stat> {
    vec![
        Stat::new("Invariant", format!("{}/{}", s.invariant, s.rounds), "").with_style(
            if s.rounds > 0 && s.invariant == s.rounds {
                CellStyle::Good
            } else {
                CellStyle::Neutral
            },
        ),
        Stat::new("Jittered", s.jittered.to_string(), "").with_style(if s.jittered == 0 {
            CellStyle::Good
        } else {
            CellStyle::Warn
        }),
        Stat::new("Collapsed", s.collapsed.to_string(), "").with_style(if s.collapsed == 0 {
            CellStyle::Good
        } else {
            CellStyle::Bad
        }),
        Stat::new("Unmeasured", s.unmeasured.to_string(), "").with_style(if s.unmeasured == 0 {
            CellStyle::Good
        } else {
            CellStyle::Bad
        }),
        // The vacuity tile. Zero means the replays never touched the restore
        // path under test, so every other tile on this row is decorative.
        Stat::new(
            "Min t1 cache",
            s.min_turn1_cached.unwrap_or(0).to_string(),
            "tok",
        )
        .with_style(if s.min_turn1_cached.unwrap_or(0) > 0 {
            CellStyle::Good
        } else {
            CellStyle::Bad
        }),
    ]
}

/// Raw gate numbers for the record. Every class is a key even when zero: a
/// missing key and a zero must stay distinguishable to whatever compares
/// records later.
pub(super) fn metrics(s: &Score) -> BTreeMap<String, f64> {
    let mut m: BTreeMap<String, f64> = [
        ("rounds", s.rounds),
        ("invariant", s.invariant),
        ("jittered", s.jittered),
        ("collapsed", s.collapsed),
        ("unmeasured", s.unmeasured),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v as f64))
    .collect();
    // The vacuity attestation, floored in BENCH.toml at `min = 1.0`. A run
    // that engaged the prefix cache and got NOTHING back records 0 here and
    // fails at the record level too — before this metric existed, such a run
    // was a green PASS that had measured nothing.
    //
    // ★ But only when a round actually reported a figure. `min_turn1_cached`
    // is `None` when no replay produced a turn-1 attestation at all, and
    // `unwrap_or(0)` wrote that absence as the vacuity value — collapsing "the
    // cache restored nothing" into "nobody looked", which is precisely the
    // distinction the doc comment above demands and the one a reader of the
    // failure ("min_cached_prompt_tokens 0 < 1") would get wrong. Both cases
    // still fail the floor: an absent key is reported as missing from the
    // record. `concurrency.rs` omits the identically-named key on `None`
    // already; this makes the two agree.
    if let Some(min) = s.min_turn1_cached {
        m.insert("min_cached_prompt_tokens".to_string(), min as f64);
    }
    m
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod report_tests;
