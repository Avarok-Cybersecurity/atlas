// SPDX-License-Identifier: AGPL-3.0-only

//! Presentation and aggregation for the MLPerf agentic leg.
//!
//! Deliberately UNLIKE `bfcl::report`, which hardcodes MLPerf-edge floors and
//! renders PASS/FAIL against them: this leg has no floors to enforce. The
//! official dataset is unpublished, so no run has ever been measured, so any
//! floor written here would be invented — and an invented floor a run can
//! clear reports PASS for something nobody measured. The verdict is therefore
//! always `info`, and the MLPerf full-dataset thresholds (inline ≥ 55.86%,
//! OSL 355–434, SWE-bench ≥ 67.5%) appear in text as CONTEXT only. The first
//! measured run on main becomes the baseline, exactly like the echolp gate's
//! history; PASS/FAIL then comes from BENCH.toml via the gate check, not from
//! this file.

use std::collections::BTreeMap;

use crate::result::{Cell, CellStyle, Column, ResultTable, Stat, Verdict};

use super::scoring::Domain;
use super::{MlperfAgentic, TurnRecord};

/// Aggregated inline scores, upstream semantics: mean over scored turns,
/// per-domain sub-scores, missing outputs at 0 in the denominator.
#[derive(Clone, Debug, Default)]
pub(super) struct Scores {
    /// Overall mean over scored turns, in [0,1].
    pub inline: f64,
    /// (mean, scored-turn count) per domain, present only if any were scored.
    pub coding: Option<(f64, usize)>,
    pub workflow: Option<(f64, usize)>,
    pub turns_scored: usize,
    /// Issued turns whose ground truth had nothing to score — excluded from
    /// the denominator, exactly as upstream lists them.
    pub turns_excluded: usize,
    /// Issued, scorable, but no model output — scored 0, IN the denominator.
    pub turns_missing: usize,
    pub output_tokens: usize,
    /// Mean output tokens per turn that produced output — the OSL axis.
    pub osl_per_turn_mean: f64,
}

/// Fold per-turn records into the run's scores. `None` when nothing was
/// scorable — the caller must treat that as a failed run, never as 0.0.
pub(super) fn aggregate(turns: &[TurnRecord]) -> Option<Scores> {
    let mut s = Scores::default();
    let (mut total, mut by_domain) = (0.0f64, BTreeMap::<&str, (f64, usize)>::new());
    let mut turns_with_output = 0usize;
    for t in turns {
        if t.missing {
            s.turns_missing += 1;
        } else {
            turns_with_output += 1;
            s.output_tokens += t.completion_tokens;
        }
        let Some(score) = t.score else {
            s.turns_excluded += 1;
            continue;
        };
        s.turns_scored += 1;
        total += score;
        let key = match t.domain {
            Domain::Coding => "coding",
            Domain::Workflow => "workflow",
        };
        let e = by_domain.entry(key).or_default();
        e.0 += score;
        e.1 += 1;
    }
    if s.turns_scored == 0 {
        return None;
    }
    s.inline = total / s.turns_scored as f64;
    s.coding = by_domain
        .get("coding")
        .map(|(sum, n)| (sum / *n as f64, *n));
    s.workflow = by_domain
        .get("workflow")
        .map(|(sum, n)| (sum / *n as f64, *n));
    s.osl_per_turn_mean = if turns_with_output > 0 {
        s.output_tokens as f64 / turns_with_output as f64
    } else {
        0.0
    };
    Some(s)
}

impl MlperfAgentic {
    pub(super) fn table(&self) -> Option<ResultTable> {
        let s = self.scores.as_ref()?;
        let mut t = ResultTable::new(
            "INLINE ACCURACY BY DOMAIN",
            vec![
                Column::left("Domain", 12),
                Column::right("turns", 8),
                Column::right("score %", 9),
            ],
        );
        for (name, entry) in [("coding", &s.coding), ("workflow", &s.workflow)] {
            if let Some((mean, n)) = entry {
                t.push(vec![
                    Cell::new(name),
                    Cell::new(n.to_string()),
                    Cell::styled(format!("{:.2}", mean * 100.0), CellStyle::Accent),
                ]);
            }
        }
        Some(t)
    }

    pub(super) fn summary(&self) -> Vec<Stat> {
        match &self.scores {
            Some(s) => vec![
                Stat::new("Inline accuracy", format!("{:.2}", s.inline * 100.0), "%")
                    .with_style(CellStyle::Accent),
                Stat::new("OSL / turn", format!("{:.1}", s.osl_per_turn_mean), "tok"),
                Stat::new("Turns scored", s.turns_scored.to_string(), ""),
                Stat::new("Missing", s.turns_missing.to_string(), "").with_style(
                    if s.turns_missing == 0 {
                        CellStyle::Good
                    } else {
                        CellStyle::Warn
                    },
                ),
            ],
            None => vec![
                Stat::new(
                    "Turns",
                    format!("{}/{}", self.cursor, self.schedule.len()),
                    "",
                ),
                Stat::new("Trajectories", self.conversations.len().to_string(), ""),
            ],
        }
    }

    pub(super) fn metrics(&self) -> BTreeMap<String, f64> {
        let Some(s) = &self.scores else {
            return BTreeMap::new();
        };
        let mut m = BTreeMap::new();
        m.insert("inline_accuracy".into(), s.inline * 100.0);
        if let Some((mean, _)) = s.coding {
            m.insert("coding_iou".into(), mean * 100.0);
        }
        if let Some((mean, _)) = s.workflow {
            m.insert("workflow_intent_acc".into(), mean * 100.0);
        }
        m.insert("osl_per_turn_mean".into(), s.osl_per_turn_mean);
        m.insert("trajectories".into(), self.conversations.len() as f64);
        m.insert("turns_scored".into(), s.turns_scored as f64);
        m.insert("turns_excluded".into(), s.turns_excluded as f64);
        m.insert("turns_missing".into(), s.turns_missing as f64);
        if let Some(wall) = self.replay_wall {
            m.insert("wall_s".into(), wall.as_secs_f64());
            if wall.as_secs_f64() > 0.0 {
                m.insert(
                    "output_tok_s".into(),
                    s.output_tokens as f64 / wall.as_secs_f64(),
                );
            }
        }
        m
    }

    /// Always `info`, never PASS — see the module doc. When a measured
    /// baseline eventually lands in BENCH.toml, requiring this gate will also
    /// need a decided verdict rule, because `check.rs` demands a PASS verdict
    /// from required gates; that decision belongs to whoever runs the first
    /// calibration, not to code written before any run existed.
    pub(super) fn verdict(&self) -> Verdict {
        let Some(s) = &self.scores else {
            return Verdict::info("not scored");
        };
        let fmt = |e: &Option<(f64, usize)>| match e {
            Some((mean, n)) => format!("{:.2}% (n={n})", mean * 100.0),
            None => "—".to_string(),
        };
        Verdict::info(format!(
            "inline {:.2}% · coding {} · workflow {} · OSL/turn {:.1} · UNMEASURED LEG: no \
             committed baseline exists; MLPerf's 55.86% / 355–434 OSL are full-dataset \
             temp-1.0 thresholds and are NOT this draw's floors. The first measured run on \
             main becomes the baseline.",
            s.inline * 100.0,
            fmt(&s.coding),
            fmt(&s.workflow),
            s.osl_per_turn_mean,
        ))
    }
}
