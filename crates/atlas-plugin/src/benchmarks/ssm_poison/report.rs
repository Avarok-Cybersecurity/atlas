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
    let mut unmeasured_seen = 0usize;
    for round in 1..=s.rounds {
        let (what, style, detail) = if let Some(turns) = collapse_map.get(&round) {
            ("COLLAPSED".to_string(), CellStyle::Bad, delta_detail(turns))
        } else if let Some(turns) = jitter_map.get(&round) {
            ("jittered".to_string(), CellStyle::Warn, delta_detail(turns))
        } else if unmeasured_seen < s.unmeasured {
            unmeasured_seen += 1;
            (
                "unmeasured".into(),
                CellStyle::Warn,
                "transport error — invariant not proven this round".into(),
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
            CellStyle::Warn
        }),
    ]
}

/// Raw gate numbers for the record. Every class is a key even when zero: a
/// missing key and a zero must stay distinguishable to whatever compares
/// records later.
pub(super) fn metrics(s: &Score) -> BTreeMap<String, f64> {
    [
        ("rounds", s.rounds),
        ("invariant", s.invariant),
        ("jittered", s.jittered),
        ("collapsed", s.collapsed),
        ("unmeasured", s.unmeasured),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v as f64))
    .collect()
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod report_tests;
