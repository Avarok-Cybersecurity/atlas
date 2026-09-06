// SPDX-License-Identifier: AGPL-3.0-only

//! How a BFCL run is presented, and what makes it pass.
//!
//! Split out of the state machine so the run logic and the reporting can each
//! be read on their own — and because the MLPerf floors belong next to the
//! verdict that enforces them, not buried in a phase loop.

use std::collections::BTreeMap;

use anyhow::Result;

use super::Bfcl;
use crate::benchmarks::bfcl::draw;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::result::{Cell, CellStyle, Column, ResultTable, Stat, Verdict};

/// MLPerf-edge `qwen3.6-27b` thresholds: the golden llama.cpp Q4_K_M reference
/// (86.23 / 87.96) x the 0.97 factor. Below these is a submission failure, not
/// a routine regression, so the verdict says so in those words.
pub const MLPERF_FLOOR_OVERALL: f64 = 83.64;
pub const MLPERF_FLOOR_NORMALIZED: f64 = 85.32;

/// The checkpoints the MLPerf-edge submission actually rides on — the ONLY
/// models whose run verdict is gated on the floors above. The floor was
/// derived from the golden llama.cpp reference for these Qwen3.6-27B weights;
/// BENCH.toml doctrine says explicitly that it does not transfer to other
/// weights, and before this scoping a healthy Qwen3.8 at 84.22/84.12 was
/// verdicted FAIL against a floor defined for a different checkpoint.
///
/// Every other checkpoint is judged by its own BENCH.toml thresholds under
/// `--pull-request-gate` (`gate::check_record`); the floor stays in the
/// summary STYLING for every model as a visual reference.
pub const MLPERF_FLOOR_CHECKPOINTS: [&str; 2] = [
    "unsloth/Qwen3.6-27B-NVFP4",
    "centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf",
];

/// Whether the served model is one of the MLPerf submission checkpoints.
/// Case-insensitive: HF ids are case-preserving but not case-sensitive, and a
/// serve that spells the org differently is still the same weights.
pub fn is_mlperf_submission_checkpoint(model: &str) -> bool {
    MLPERF_FLOOR_CHECKPOINTS
        .iter()
        .any(|c| c.eq_ignore_ascii_case(model))
}

/// The baseline floors a NON-MLPerf checkpoint's run verdict self-judges
/// against — the `threshold_params` pair the gate auto-fills from the served
/// variant's own BENCH.toml `min` bounds (descriptors.rs). 0.0 on both means
/// not gating: a standalone run keeps the info verdict, because inventing a
/// bar for weights with no committed baseline is the miscomparison the MLPerf
/// scoping removed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct BaselineMins {
    pub overall: f64,
    pub normalized: f64,
}

impl BaselineMins {
    /// The two schema params, appended to the driver's `parameters()`.
    // 0.0 default is the documented OFF state, not an implicit bar (PCND):
    // standalone runs must stay info-verdicted, and the gate substitutes the
    // real floors per variant.
    pub(super) fn specs() -> [ParamSpec; 2] {
        const PCT: ParamKind = ParamKind::Float {
            min: 0.0,
            max: 100.0,
        };
        [
            ParamSpec::new(
                "min_overall",
                "Overall floor",
                "Run-verdict floor on overall_accuracy. 0 disables (a standalone run reports \
                 an info verdict); under --pull-request-gate this is auto-filled from the \
                 selected variant's BENCH.toml `min` bound. Ignored on the MLPerf submission \
                 checkpoints, which keep the MLPerf floor verdict.",
                PCT,
                ParamValue::Float(0.0),
            ),
            ParamSpec::new(
                "min_normalized",
                "Normalized floor",
                "Run-verdict floor on normalized_single_turn_score. 0 disables (a standalone \
                 run reports an info verdict); under --pull-request-gate this is auto-filled \
                 from the selected variant's BENCH.toml `min` bound. Ignored on the MLPerf \
                 submission checkpoints, which keep the MLPerf floor verdict.",
                PCT,
                ParamValue::Float(0.0),
            ),
        ]
    }

    pub(super) fn from_values(values: &ParamValues) -> Result<Self> {
        Ok(Self {
            overall: values.float("min_overall")?,
            normalized: values.float("min_normalized")?,
        })
    }

    fn gating(self) -> bool {
        self.overall > 0.0 || self.normalized > 0.0
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
            Some(s) => vec![
                Stat::new(
                    "Overall accuracy",
                    format!("{:.2}", s.overall_accuracy),
                    "%",
                )
                .with_style(floor_style(s.overall_accuracy, MLPERF_FLOOR_OVERALL)),
                Stat::new(
                    "Normalized single-turn",
                    format!("{:.2}", s.normalized_single_turn_score),
                    "%",
                )
                .with_style(floor_style(
                    s.normalized_single_turn_score,
                    MLPERF_FLOOR_NORMALIZED,
                )),
                Stat::new("Samples", s.total_samples.to_string(), ""),
            ],
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
        // Per-subset tallies, flattened into the metrics map so a shard's record
        // carries everything the group aggregate needs and no record schema
        // changes. `check_record` iterates the ENTRY's thresholds and looks each
        // up here, so metrics nothing gates are ignored.
        //
        // Counts, not scores: score.py weights hierarchically, so a mean of
        // shard scores is not the whole-set value. See `aggregate`.
        for (subset, (hits, n)) in &s.subset_totals {
            m.insert(format!("subset.{subset}.hits"), *hits as f64);
            m.insert(format!("subset.{subset}.n"), *n as f64);
        }
        m
    }

    pub(super) fn verdict(&self) -> Verdict {
        let Some(s) = &self.scores else {
            return Verdict::info("not scored");
        };
        floor_verdict(self.target_model.as_deref(), s, self.baseline_mins)
    }
}

/// The run-level verdict, scoped by served model.
///
/// The MLPerf floor FAILS a run only on the submission checkpoints it was
/// derived for. Any other checkpoint is judged against ITS OWN committed bars
/// when the gate (or an operator) supplied them via `mins` — a PASS/FAIL the
/// gate machinery can consume, since `GateRecord::verdict_passes` accepts
/// nothing short of PASS. Without bars it gets an INFO verdict that names its
/// real judge instead of failing on an alien floor. An unknown model (no
/// `load()` happened) is treated the same way: applying the MLPerf floor to
/// weights nobody identified is exactly the miscomparison this scoping
/// removes.
///
/// Pure so the scoping is unit-testable without a live endpoint.
fn floor_verdict(target_model: Option<&str>, s: &super::Scores, mins: BaselineMins) -> Verdict {
    let detail = format!(
        "overall {:.2} (floor {MLPERF_FLOOR_OVERALL}) · normalized {:.2} (floor \
         {MLPERF_FLOOR_NORMALIZED}) · n={}",
        s.overall_accuracy, s.normalized_single_turn_score, s.total_samples
    );
    match target_model {
        Some(m) if is_mlperf_submission_checkpoint(m) => {
            let overall_ok = s.overall_accuracy >= MLPERF_FLOOR_OVERALL;
            let normalized_ok = s.normalized_single_turn_score >= MLPERF_FLOOR_NORMALIZED;
            if overall_ok && normalized_ok {
                Verdict::pass(detail)
            } else {
                Verdict::fail(format!("BELOW THE MLPERF-EDGE FLOOR — {detail}"))
            }
        }
        // ★ Deliberately STRICTER than gate scoring: this compares the raw
        // value >= min, while `gate::scoring` allows value + noise >= min. A
        // sub-noise dip therefore fails the run verdict even though scoring
        // would have passed it — safe conservatism (it can only re-run a
        // healthy build, never green-light a regression), and it keeps the
        // driver free of a second copy of the noise model.
        _ if mins.gating() => {
            let bars = format!(
                "overall {:.2} (baseline min {:.2}) · normalized {:.2} (baseline min {:.2}) \
                 · n={}",
                s.overall_accuracy,
                mins.overall,
                s.normalized_single_turn_score,
                mins.normalized,
                s.total_samples
            );
            if s.overall_accuracy >= mins.overall
                && s.normalized_single_turn_score >= mins.normalized
            {
                Verdict::pass(format!("{bars} — clears this checkpoint's committed bars"))
            } else {
                Verdict::fail(format!("BELOW THE BASELINE THRESHOLDS — {bars}"))
            }
        }
        Some(m) => Verdict::info(format!(
            "{detail} — judged by baseline thresholds: {m} is not an MLPerf submission \
             checkpoint, so the floor does not gate this run (it does not transfer \
             across weights); the floor styling above is a visual reference only"
        )),
        None => Verdict::info(format!(
            "{detail} — judged by baseline thresholds: served model unknown, so the \
             MLPerf floor (defined on the Qwen3.6-27B submission checkpoints) does \
             not gate this run"
        )),
    }
}

fn floor_style(value: f64, floor: f64) -> CellStyle {
    if value >= floor {
        CellStyle::Good
    } else {
        CellStyle::Bad
    }
}
