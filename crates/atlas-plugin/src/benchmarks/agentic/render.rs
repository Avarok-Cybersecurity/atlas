// SPDX-License-Identifier: AGPL-3.0-only

//! How a finished `agentic-webserver` tier is PRESENTED — the per-iteration
//! table and the summary tiles.
//!
//! Split out of `mod.rs` for the repo's 500-line ceiling, the same reason
//! `score.rs` and `params.rs` were. Presentation is the natural seam: nothing
//! here decides anything. Both functions read the SAME aggregates the verdict
//! and the record read, so a tile can never disagree with the pass/fail line.

use super::AgenticWebserver;
use crate::result::{Cell, CellStyle, Column, ResultTable, Stat};

impl AgenticWebserver {
    pub(super) fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            "ITERATIONS",
            vec![
                Column::right("Run", 4),
                Column::right("wall s", 8),
                Column::left("ws_ok", 6),
                Column::right("steps", 6),
                Column::right("turns", 6),
                Column::right("tools", 6),
                Column::left("note", 40),
            ],
        );
        for r in &self.rows {
            t.push(vec![
                Cell::new(r.index.to_string()),
                Cell::new(format!("{:.1}", r.wall_s)),
                Cell::styled(
                    if r.webserver_ok { "pass" } else { "FAIL" },
                    if r.webserver_ok {
                        CellStyle::Good
                    } else {
                        CellStyle::Bad
                    },
                ),
                Cell::styled(
                    format!("{}/6", r.directions.met()),
                    if r.directions.overall() {
                        CellStyle::Good
                    } else {
                        CellStyle::Warn
                    },
                ),
                Cell::new(r.turns.to_string()),
                Cell::new(r.tool_calls.to_string()),
                Cell::styled(r.note.clone(), CellStyle::Dim),
            ]);
        }
        t
    }

    pub(super) fn summary(&self) -> Vec<Stat> {
        let ok = self.rows.iter().filter(|r| r.webserver_ok).count();
        let fd = self.rows.iter().filter(|r| r.directions.overall()).count();
        let n = self.rows.len();
        vec![
            Stat::new("webserver_ok", format!("{ok}/{n}"), "").with_style(if n > 0 && ok == n {
                CellStyle::Good
            } else {
                CellStyle::Warn
            }),
            Stat::new("followed_directions", format!("{fd}/{n}"), "").with_style(
                if n > 0 && fd == n {
                    CellStyle::Good
                } else {
                    CellStyle::Warn
                },
            ),
            Stat::new(
                "s/turn",
                self.seconds_per_turn()
                    .map_or_else(|| "n/a".to_string(), |s| format!("{s:.3}")),
                "s",
            )
            .with_style(match self.seconds_per_turn() {
                // A tier that took no turns is not fast, it is broken; a green
                // speed cell reads as a pass at a glance.
                None => CellStyle::Warn,
                // Neutral for an unbounded variant: nothing to be good or bad
                // against until one commits a measured bound.
                Some(_) if self.s_per_turn_budget <= 0.0 => CellStyle::Neutral,
                Some(s) if s <= self.s_per_turn_budget => CellStyle::Good,
                Some(_) => CellStyle::Warn,
            }),
            Stat::new("Σ wall", format!("{:.0}", self.total_wall()), "s").with_style(
                if self.total_wall() <= self.wall_budget_s {
                    CellStyle::Good
                } else {
                    CellStyle::Warn
                },
            ),
        ]
    }
}
