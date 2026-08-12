// SPDX-License-Identifier: AGPL-3.0-only

//! How a poisoning run is presented. Pure functions of the [`super::score::Score`],
//! the same split `contamination/report.rs` uses: run logic and rendering read
//! separately, and everything here is table-testable with no server.

use std::collections::BTreeMap;

use super::score::Score;
use crate::result::{Cell, CellStyle, Column, ResultTable, Stat};

/// One row per replay round, in round order.
pub(super) fn table(s: &Score) -> ResultTable {
    let mut t = ResultTable::new(
        "REPLAY ROUNDS",
        vec![
            Column::left("Round", 6),
            Column::left("Result", 20),
            Column::left("Detail", 44),
        ],
    );
    let diverged_map: BTreeMap<usize, &Vec<usize>> = s
        .diverged_rounds
        .iter()
        .map(|(n, turns)| (*n, turns))
        .collect();
    let mut unmeasured_seen = 0usize;
    for round in 1..=s.rounds {
        let (what, style, detail) = if let Some(turns) = diverged_map.get(&round) {
            (
                format!("DIVERGED (turns {:?})", turns),
                CellStyle::Bad,
                "output differed from the reference for this round".into(),
            )
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

/// The headline tiles. `Invariant x/y` is Good only when every replay held.
pub(super) fn summary(s: &Score) -> Vec<Stat> {
    vec![
        Stat::new("Invariant", format!("{}/{}", s.invariant, s.rounds), "").with_style(
            if s.rounds > 0 && s.invariant == s.rounds {
                CellStyle::Good
            } else {
                CellStyle::Bad
            },
        ),
        Stat::new("Diverged", s.diverged.to_string(), "").with_style(if s.diverged == 0 {
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
        ("diverged", s.diverged),
        ("unmeasured", s.unmeasured),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v as f64))
    .collect()
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod report_tests;
