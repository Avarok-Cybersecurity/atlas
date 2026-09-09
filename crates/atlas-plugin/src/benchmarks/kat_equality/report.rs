// SPDX-License-Identifier: AGPL-3.0-only

//! How an equality run is presented. Pure functions of [`super::compare`] —
//! no server, no I/O, so every row is table-testable.

use std::collections::BTreeMap;

use super::compare::{OrderRun, SampleVerdict, Score, verdict_for};
use crate::result::{Cell, CellStyle, Column, ResultTable, Stat};

pub(super) fn summary(s: &Score) -> Vec<Stat> {
    vec![
        Stat::new("orders", s.orders.to_string(), ""),
        Stat::new("samples", s.samples.to_string(), ""),
        Stat::new("identical", s.equal.to_string(), ""),
        Stat::new("DIVERGED", s.diverged.to_string(), ""),
    ]
}

/// One row per sample that is NOT equal — plus, when everything agreed, a
/// single row saying so.
///
/// Deliberately not one row per sample: a 995-row table where 995 rows say
/// "identical" buries the four that do not. The failures ARE the report.
pub(super) fn table(s: &Score, runs: &[OrderRun]) -> ResultTable {
    let mut t = ResultTable::new(
        "SAMPLES THAT CHANGED WITH THE ORDER",
        vec![
            Column::left("Sample", 34),
            Column::left("Result", 12),
            Column::left("Detail", 46),
        ],
    );
    let Some(reference) = runs.first() else {
        return t;
    };
    let mut shown = 0usize;
    for obs in &reference.observations {
        let v = verdict_for(&obs.sample_id, reference, &runs[1..]);
        let (what, style, detail) = match v {
            SampleVerdict::Equal => continue,
            SampleVerdict::Diverged {
                other_order,
                common_prefix,
            } => (
                "DIVERGED",
                CellStyle::Bad,
                format!("differs from `{other_order}` after {common_prefix} identical bytes"),
            ),
            SampleVerdict::Unmeasured(reason) => ("unmeasured", CellStyle::Bad, reason),
        };
        t.push(vec![
            Cell::new(obs.sample_id.clone()),
            Cell::styled(what.to_string(), style),
            Cell::new(detail),
        ]);
        shown += 1;
        // A verdict that scrolls is a verdict nobody reads to the end of.
        if shown >= 40 {
            let left = s.diverged + s.unmeasured - shown;
            if left > 0 {
                t.push(vec![
                    Cell::new(format!("… {left} more")),
                    Cell::new(String::new()),
                    Cell::new("see the per-sample metrics".to_string()),
                ]);
            }
            break;
        }
    }
    if shown == 0 {
        t.push(vec![
            Cell::styled("all samples".to_string(), CellStyle::Good),
            Cell::styled("identical".to_string(), CellStyle::Good),
            Cell::new(format!("byte-for-byte across {} request orders", s.orders)),
        ]);
    }
    t
}

/// Raw gate numbers. Every class is a key even at zero: a missing key and a
/// zero must stay distinguishable to whatever compares records later.
pub(super) fn metrics(s: &Score) -> BTreeMap<String, f64> {
    [
        ("orders", s.orders),
        ("samples", s.samples),
        ("identical", s.equal),
        ("diverged", s.diverged),
        ("unmeasured", s.unmeasured),
        // The vacuity attestation, and the reason a bound on it belongs in
        // BENCH.toml: a run whose replies were all empty agrees perfectly and
        // has measured nothing. `diverged = 0` alone is not evidence.
        ("empty_replies", s.empty_replies),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v as f64))
    .collect()
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod report_tests;
