// SPDX-License-Identifier: AGPL-3.0-only

//! Rendering the matrix: one row per round, one tile row of headline numbers.
//!
//! Every row states its own outcome per signal, so a failure is attributable to
//! a model × quant × signal without going back to the log — which is the whole
//! reason the Python harness's per-model JSON existed.

use super::plan::Plan;
use super::score::{Outcome, RoundResult, Signal, Tally};
use crate::result::{Cell, CellStyle, Column, Stat};
use crate::result::{ResultTable, Verdict};

fn style_of(signal: &Signal) -> CellStyle {
    match signal {
        Signal::Pass => CellStyle::Good,
        Signal::Fail(_) => CellStyle::Bad,
        Signal::NotApplicable(_) => CellStyle::Warn,
        Signal::NotRun => CellStyle::Dim,
    }
}

fn signal_cell(signal: &Signal) -> Cell {
    Cell::styled(signal.text(), style_of(signal))
}

/// The full matrix, including the rounds that were skipped and why.
pub fn table(plan: &Plan, results: &[RoundResult]) -> ResultTable {
    let mut t = ResultTable::new(
        "SERVE MATRIX",
        vec![
            Column::left("Model × quant", 40),
            Column::left("Serving", 8),
            Column::left("Coherence", 10),
            Column::left("Codegen", 8),
            Column::left("Tools", 6),
            Column::left("Long ctx", 9),
            Column::right("tok/s", 8),
            Column::left("Verdict", 28),
        ],
    );
    for round in &plan.rounds {
        let label = round.label();
        // A skipped checkpoint is a ROW, not an omission. Leaving it out is how
        // "8/8 verified" comes to mean something other than what it says.
        if let Some(why) = round.skipped {
            t.push(skipped_row(&label, why.reason()));
            continue;
        }
        if round.excluded {
            t.push(skipped_row(&label, "outside the filter"));
            continue;
        }
        let found = results.iter().find(|r| r.label == label);
        match found {
            Some(r) => t.push(result_row(&label, r)),
            None => t.push(missing_result_row(&label)),
        }
    }
    t
}

fn skipped_row(label: &str, why: &str) -> Vec<Cell> {
    let mut row = vec![Cell::new(label)];
    row.extend((0..6).map(|_| Cell::styled("—", CellStyle::Dim)));
    row.push(Cell::styled(format!("SKIP · {why}"), CellStyle::Dim));
    row
}

fn missing_result_row(label: &str) -> Vec<Cell> {
    let mut row = vec![Cell::new(label)];
    row.extend((0..6).map(|_| Cell::styled("—", CellStyle::Dim)));
    row.push(Cell::styled(
        "FAIL · no result — did not run",
        CellStyle::Bad,
    ));
    row
}

fn result_row(label: &str, r: &RoundResult) -> Vec<Cell> {
    let bars = r.bars();
    let verdict = if bars.is_empty() {
        Cell::styled("PASS", CellStyle::Good)
    } else {
        Cell::styled(format!("FAIL · {}", bars.join(", ")), CellStyle::Bad)
    };
    let Some(s) = r.signals() else {
        let mut row = vec![Cell::new(label)];
        row.extend((0..6).map(|_| Cell::styled("—", CellStyle::Dim)));
        row.push(verdict);
        return row;
    };
    let coherence = format!("{}/{}", s.coherence_pass, s.coherence_total);
    let coherence_style = if s.coherence_pass == s.coherence_total {
        CellStyle::Good
    } else {
        CellStyle::Bad
    };
    let tps = match s.tps {
        // "no baseline" belongs beside the number, not in a footnote: the bar
        // it would have been checked against does not exist.
        Some(v) if v > 0.0 => Cell::styled(format!("{v:.1}"), CellStyle::Accent),
        Some(_) => Cell::styled("0", CellStyle::Bad),
        None => Cell::styled("—", CellStyle::Dim),
    };
    vec![
        Cell::new(label),
        // Is this row's numbers' model the one the label names? Its own
        // column, because every other cell is meaningless when it is not.
        signal_cell(&s.identity),
        Cell::styled(coherence, coherence_style),
        signal_cell(&s.codegen),
        signal_cell(&s.tool_call),
        signal_cell(&s.long_ctx),
        tps,
        verdict,
    ]
}

pub fn summary(tally: &Tally) -> Vec<Stat> {
    vec![
        Stat::new(
            "Verified",
            format!("{}/{}", tally.verified, tally.planned),
            "planned",
        )
        // 0/0 is "nothing was measured", which the verdict renders as Info —
        // a red tile beside it would contradict it.
        .with_style(match (tally.planned, tally.passed()) {
            (0, _) => CellStyle::Dim,
            (_, true) => CellStyle::Good,
            (_, false) => CellStyle::Bad,
        }),
        Stat::new("Below bar", tally.failures.len().to_string(), "").with_style(
            if tally.failures.is_empty() {
                CellStyle::Dim
            } else {
                CellStyle::Bad
            },
        ),
        Stat::new("Not runnable", tally.skipped.to_string(), "on this box")
            .with_style(CellStyle::Dim),
        Stat::new("Filtered out", tally.excluded.to_string(), "").with_style(CellStyle::Dim),
    ]
}

/// PASS/FAIL, or an honest `Info` when there was nothing to gate.
pub fn verdict(tally: &Tally, plan: &Plan) -> Verdict {
    let text = super::score::verdict_text(tally, plan);
    if tally.planned == 0 {
        return Verdict::info(format!(
            "no checkpoint on this box is runnable — nothing was measured. {text}"
        ));
    }
    if tally.passed() {
        Verdict::pass(text)
    } else {
        Verdict::fail(text)
    }
}

/// The per-round log line: enough to attribute a failure without the table.
pub fn round_line(r: &RoundResult) -> String {
    match &r.outcome {
        Outcome::BootFailed(why) => format!("{}: DID NOT BOOT — {why}", r.label),
        Outcome::NotReached => format!("{}: not reached", r.label),
        Outcome::Probed(s) => {
            let bars = r.bars();
            let mut line = format!(
                "{}: serving {} · coherence {}/{} · codegen {} · tools {} · long-ctx {} · {}",
                r.label,
                s.identity.text(),
                s.coherence_pass,
                s.coherence_total,
                s.codegen.text(),
                s.tool_call.text(),
                s.long_ctx.text(),
                match s.tps {
                    Some(v) => format!("{v:.1} tok/s"),
                    None => "tok/s unmeasured (one SSE delta)".into(),
                }
            );
            if let Some(note) = r.tps_note() {
                line.push_str(&format!(" ({note})"));
            }
            if !bars.is_empty() {
                line.push_str(&format!(" → FAIL {}", bars.join(", ")));
                // The reason the probe gave, so the line is enough on its own.
                // "codegen FAIL" without it sends the reader back to a log that
                // no longer holds the reply.
                let why: Vec<&str> = [&s.identity, &s.codegen, &s.tool_call, &s.long_ctx]
                    .iter()
                    .filter(|sig| sig.is_fail())
                    .filter_map(|sig| sig.detail())
                    .collect();
                if !why.is_empty() {
                    line.push_str(&format!(" — {}", why.join("; ")));
                }
            }
            line
        }
    }
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
