// SPDX-License-Identifier: AGPL-3.0-only

//! How a BFCL run is presented, and what makes it pass.
//!
//! Split out of the state machine so the run logic and the reporting can each
//! be read on their own — and because the floors belong next to the verdict
//! that enforces them, not buried in a phase loop. The floors are
//! per-variant: scoring the echolp draw against the golden MLPerf-edge
//! numbers is the bug this split exists to prevent.

use std::collections::BTreeMap;

use super::{Bfcl, Variant};
use crate::benchmarks::bfcl::draw;
use crate::result::{Cell, CellStyle, Column, ResultTable, Stat, Verdict};

/// MLPerf-edge `qwen3.6-27b` thresholds: the golden llama.cpp Q4_K_M reference
/// (86.23 / 87.96) x the 0.97 factor. Below these is a submission failure, not
/// a routine regression, so the verdict says so in those words.
///
/// These apply to the golden n=995 draw (`bfcl-subset` / `bfcl-full`) only.
pub const MLPERF_FLOOR_OVERALL: f64 = 83.64;
pub const MLPERF_FLOOR_NORMALIZED: f64 = 85.32;

/// Echolp-draw floors: the 35B high-water ratchet (84.66 / 83.32). Locked to
/// `kernels/gb10/qwen3.6-35b-a3b/BENCH.toml` by test. Do not use the
/// MLPerf-edge pair — the category mix moves normalized by ~1.8 points.
pub const ECHOLP_FLOOR_OVERALL: f64 = 84.66;
pub const ECHOLP_FLOOR_NORMALIZED: f64 = 83.32;

/// The two accuracy floors a variant's runtime verdict applies, and the name
/// the fail reason prints so a crossed-draw miss is readable.
pub(super) struct ScoreFloors {
    pub overall: f64,
    pub normalized: f64,
    pub name: &'static str,
}

impl Variant {
    pub(super) fn score_floors(self) -> ScoreFloors {
        match self {
            Self::Subset | Self::Full => ScoreFloors {
                overall: MLPERF_FLOOR_OVERALL,
                normalized: MLPERF_FLOOR_NORMALIZED,
                name: "MLPERF-EDGE",
            },
            Self::SubsetEcholp => ScoreFloors {
                overall: ECHOLP_FLOOR_OVERALL,
                normalized: ECHOLP_FLOOR_NORMALIZED,
                name: "ECHOLP",
            },
        }
    }
}

impl Bfcl {
    pub(super) fn table(&self) -> Option<ResultTable> {
        let scores = self.scores.as_ref()?;
        let mut t = ResultTable::new(
            "PER-SUBSET ACCURACY",
            vec![
                Column::left("Subset", 24),
                Column::left("Category", 14),
                Column::right("accuracy %", 11),
            ],
        );
        for (subset, value) in &scores.subset_scores {
            t.push(vec![
                Cell::new(subset.clone()),
                Cell::styled(
                    draw::category_of(subset).unwrap_or("unscored").to_string(),
                    CellStyle::Dim,
                ),
                Cell::styled(
                    format!("{value:.2}"),
                    match *value {
                        v if v >= 90.0 => CellStyle::Good,
                        v if v >= 60.0 => CellStyle::Neutral,
                        _ => CellStyle::Warn,
                    },
                ),
            ]);
        }
        for (category, value) in &scores.category_scores {
            t.push(vec![
                Cell::styled(format!("▸ {category}"), CellStyle::Accent),
                Cell::styled("category".to_string(), CellStyle::Dim),
                Cell::styled(format!("{value:.2}"), CellStyle::Accent),
            ]);
        }
        Some(t)
    }

    pub(super) fn summary(&self) -> Vec<Stat> {
        match &self.scores {
            Some(s) => {
                let floors = self.variant.score_floors();
                vec![
                    Stat::new(
                        "Overall accuracy",
                        format!("{:.2}", s.overall_accuracy),
                        "%",
                    )
                    .with_style(floor_style(s.overall_accuracy, floors.overall)),
                    Stat::new(
                        "Normalized single-turn",
                        format!("{:.2}", s.normalized_single_turn_score),
                        "%",
                    )
                    .with_style(floor_style(
                        s.normalized_single_turn_score,
                        floors.normalized,
                    )),
                    Stat::new("Samples", s.total_samples.to_string(), ""),
                ]
            }
            None => vec![
                Stat::new(
                    "Samples",
                    format!("{}/{}", self.cursor, self.samples.len()),
                    "",
                ),
                Stat::new("With tool calls", self.tool_call_samples.to_string(), ""),
            ],
        }
    }

    /// Raw gate numbers for `--pull-request-gate` (same source the summary
    /// tiles read from). Empty until scoring completes.
    pub(super) fn metrics(&self) -> BTreeMap<String, f64> {
        let Some(s) = &self.scores else {
            return BTreeMap::new();
        };
        let mut m = BTreeMap::new();
        m.insert("overall_accuracy".to_string(), s.overall_accuracy);
        m.insert(
            "normalized_single_turn_score".to_string(),
            s.normalized_single_turn_score,
        );
        m.insert("samples".to_string(), s.total_samples as f64);
        m
    }

    pub(super) fn verdict(&self) -> Verdict {
        let Some(s) = &self.scores else {
            return Verdict::info("not scored");
        };
        let floors = self.variant.score_floors();
        let overall_ok = s.overall_accuracy >= floors.overall;
        let normalized_ok = s.normalized_single_turn_score >= floors.normalized;
        let detail = format!(
            "overall {:.2} (floor {}) · normalized {:.2} (floor {}) · n={}",
            s.overall_accuracy,
            floors.overall,
            s.normalized_single_turn_score,
            floors.normalized,
            s.total_samples
        );
        if overall_ok && normalized_ok {
            Verdict::pass(detail)
        } else {
            Verdict::fail(format!("BELOW THE {} FLOOR — {detail}", floors.name))
        }
    }
}

fn floor_style(value: f64, floor: f64) -> CellStyle {
    if value >= floor {
        CellStyle::Good
    } else {
        CellStyle::Bad
    }
}
