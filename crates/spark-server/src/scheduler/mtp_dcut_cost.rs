// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08), vllm/v1/worker/gpu/spec_decode/adaptive_verification.py (build_cost_tables_from_curves, lines 163-213; get_num_tokens, lines 399-469), Copyright contributors to the vLLM project, Apache-2.0.
// Changes: the cost table is learned online (EWMA of measured round wall per verify row count) instead of profiled; the budget is chosen per verify batch on the host from the calibrated survival scores; every sequence keeps at least one draft and the width cap stays. See DCUT-V2-ATTRIBUTION.md.

//! Responsibility: D-Cut v2, the cost-optimal verify budget (`mtp_dcut_cost_model`).
//!
//! D-Cut v1 keeps a FIXED fraction of the batch's optional draft rows. v2 keeps
//! the number that maximizes expected tokens per second:
//!
//! ```text
//! B* = argmax_B  (n + sum_i s_i0 + sum of the top-B optional survivals)
//!                / cost(2n + B)
//! ```
//!
//! `n` sequences each contribute a base row and a mandatory first draft (the
//! v1 envelope: `rows_i` stays in 2..=4), the optional slots are every draft at
//! depth 2 and beyond ranked by calibrated survival ([`super::estimator`]), and
//! `cost(R)` is an online table: an EWMA of the measured wall time of a batched
//! MTP round that verified `R` rows. The draft cost is constant in `R` (the
//! drafter has already run at full width), so it is part of every table entry.
//!
//! # Table rules
//!
//! * An entry counts only after [`MIN_SAMPLES`] observations. Calibrated points
//!   are made monotone (running max, as upstream) and linearly interpolated;
//!   below the lowest point the lowest cost is used (an over-estimate of the
//!   saving, so pessimistic about pruning); above the highest there is no
//!   estimate, so the planner is NOT calibrated for that width and D-Cut v1's
//!   fixed ratio runs instead.
//! * While not calibrated, every [`WARMUP_PROBE_PERIOD`]th plan keeps the full
//!   width (a probe) so the top of the table fills; once calibrated, every
//!   [`REPROBE_PERIOD`]th plan does, so its EWMA follows the workload.
//! * Only a round where every sequence went through the batched plan (no
//!   bootstrap, no serial leftover, one chunk) feeds the table, so the row axis
//!   describes one kind of forward.
//!
//! The row is OFF on every target; with it off nothing here runs ([`decide`]
//! answers [`Decision::Legacy`] and the feeds return at once) and the planner is
//! byte for byte D-Cut v1. Accept and reject stay exact either way: the model
//! only chooses how many drafts to verify.

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use super::estimator::{AcceptEstimator, MAX_POS};
use super::{VERIFY_ROW_BUDGET, WIDTH_CAP};
use crate::api::port_metrics as pm;
use crate::scheduler::types::ActiveSeq;

/// Observations a table entry needs before it counts.
pub const MIN_SAMPLES: u32 = 8;
const EWMA_ALPHA: f64 = 0.2;
/// While the top of the table is empty, every this-many-th plan probes full width.
const WARMUP_PROBE_PERIOD: u64 = 4;
/// Once calibrated, every this-many-th plan probes full width to refresh the top.
const REPROBE_PERIOD: u64 = 128;

/// Is the `mtp_dcut_cost_model` row on (`AVAROK_MTP_DCUT_COST`)? Read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(spark_model::layers::ops::port_rows::mtp_dcut_cost_model::row)
}

/// EWMA of a round's wall time per total verify row count.
#[derive(Debug, Clone)]
pub struct CostTable {
    ewma: Vec<f64>,
    n: Vec<u32>,
}

impl Default for CostTable {
    fn default() -> Self {
        Self::new()
    }
}

impl CostTable {
    pub fn new() -> Self {
        Self {
            ewma: vec![0.0; VERIFY_ROW_BUDGET + 1],
            n: vec![0; VERIFY_ROW_BUDGET + 1],
        }
    }

    /// Fold one measurement. Out-of-range rows and non-positive times are ignored.
    pub fn observe(&mut self, rows: usize, wall_us: f64) {
        if rows == 0 || rows >= self.ewma.len() || !wall_us.is_finite() || wall_us <= 0.0 {
            return;
        }
        if self.n[rows] == 0 {
            self.ewma[rows] = wall_us;
        } else {
            self.ewma[rows] += EWMA_ALPHA * (wall_us - self.ewma[rows]);
        }
        self.n[rows] = self.n[rows].saturating_add(1);
    }

    /// Calibrated `(rows, cost)` points, cost made monotone by a running max.
    fn points(&self) -> Vec<(usize, f64)> {
        let mut run = 0.0f64;
        let mut out = Vec::new();
        for (r, (&c, &n)) in self.ewma.iter().zip(&self.n).enumerate() {
            if n >= MIN_SAMPLES {
                run = run.max(c);
                out.push((r, run));
            }
        }
        out
    }

    /// True when a calibrated point exists at or above `hi` rows.
    pub fn covers(&self, hi: usize) -> bool {
        self.points().last().is_some_and(|&(r, _)| r >= hi)
    }

    /// Cost of every row count in `lo..=hi`, or `None` when the table does not
    /// cover `hi` (see the module docs for the interpolation rules).
    pub fn curve(&self, lo: usize, hi: usize) -> Option<Vec<f64>> {
        let pts = self.points();
        if lo > hi || pts.last()?.0 < hi {
            return None;
        }
        Some(
            (lo..=hi)
                .map(|r| {
                    let i = pts.partition_point(|p| p.0 <= r);
                    if i == 0 {
                        return pts[0].1;
                    }
                    let (r0, c0) = pts[i - 1];
                    match pts.get(i) {
                        Some(&(r1, c1)) if r0 != r => {
                            c0 + (c1 - c0) * (r - r0) as f64 / (r1 - r0) as f64
                        }
                        _ => c0,
                    }
                })
                .collect(),
        )
    }
}

/// `argmax_b (base_tokens + sum of the b largest survivals) / costs[b]`;
/// `costs[b]` is the round cost with `b` optional drafts, so
/// `costs.len() == survival_desc.len() + 1`. Ties go to the LARGER `b`, so a
/// flat cost table selects the full budget.
pub fn optimal_budget(base_tokens: f64, survival_desc: &[f64], costs: &[f64]) -> usize {
    let (mut best, mut best_v, mut tokens) = (0usize, f64::NEG_INFINITY, base_tokens);
    for (b, &c) in costs.iter().enumerate() {
        if b > 0 {
            tokens += survival_desc.get(b - 1).copied().unwrap_or(0.0);
        }
        let v = tokens / c.max(1e-9);
        if v >= best_v {
            (best, best_v) = (b, v);
        }
    }
    best
}

/// What the planner should do for one verify batch.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Row off: D-Cut v1 exactly.
    Legacy,
    /// Not calibrated: D-Cut v1 at this ratio.
    Warmup(f32),
    /// Keep the full width this plan to fill or refresh the table.
    Probe,
    /// The cost-optimal retained draft count per sequence (`1..=k`).
    Retained(Vec<usize>),
}

impl Decision {
    fn tag(&self) -> usize {
        match self {
            Self::Legacy => 0,
            Self::Warmup(_) => 1,
            Self::Probe => 2,
            Self::Retained(_) => 3,
        }
    }
}

/// The online cost table, the acceptance estimator and the planner state.
#[derive(Debug, Clone, Default)]
pub struct CostModel {
    pub table: CostTable,
    pub est: AcceptEstimator,
    plans: u64,
    last_fed: (usize, u64),
}

impl CostModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one batched round: `rows` verified, `wall_us` of wall time. A call
    /// identical to the previous one is ignored (a second feeder of the same
    /// round must not double its weight). Counts one estimator tick.
    pub fn observe_verify(&mut self, rows: usize, wall_us: u64) {
        if (rows, wall_us) == self.last_fed {
            return;
        }
        self.last_fed = (rows, wall_us);
        self.table.observe(rows, wall_us as f64);
        self.est.tick();
    }

    /// 3.7.4 `budget`: the number of optional draft rows to keep out of
    /// `survival_desc` (survival scores, any order; sorted here), where
    /// `non_draft_rows` rows are present regardless and each yields one
    /// guaranteed token. `None` while the table does not cover the range.
    pub fn budget(&self, survival_desc: &[f32], non_draft_rows: usize) -> Option<usize> {
        let mut surv: Vec<f64> = survival_desc.iter().map(|&s| f64::from(s)).collect();
        surv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let costs = self
            .table
            .curve(non_draft_rows, non_draft_rows + surv.len())?;
        Some(optimal_budget(non_draft_rows as f64, &surv, &costs))
    }

    /// Plan one verify batch of `confs.len()` sequences at depth `k >= 2`.
    /// `confs[i]` is sequence i's drafter top-1 log-probabilities; `base_ratio`
    /// is D-Cut v1's ratio, used until the table is calibrated.
    pub fn decide(
        &mut self,
        confs: &[&[f32]],
        k: usize,
        row_budget: usize,
        base_ratio: f32,
    ) -> Decision {
        self.plans += 1;
        let n = confs.len();
        let base_rows = 2 * n;
        let max_extra = (n * (k - 1)).min(row_budget.saturating_sub(base_rows));
        let covered = self.table.covers(base_rows + max_extra);
        let period = if covered {
            REPROBE_PERIOD
        } else {
            WARMUP_PROBE_PERIOD
        };
        if self.plans.is_multiple_of(period) {
            return Decision::Probe;
        }
        if !covered {
            return Decision::Warmup(base_ratio);
        }
        let mut slots: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (k - 1));
        let mut mandatory = 0.0;
        for (i, c) in confs.iter().enumerate() {
            let s = self.est.survival(c, k);
            mandatory += s[0];
            slots.extend(s.iter().enumerate().skip(1).map(|(j, &sj)| (sj, i, j)));
        }
        // Descending survival; ties on (sequence, depth) keep the set a pure
        // function of the batch and prefix-closed (survival never rises with depth).
        slots.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        });
        slots.truncate(max_extra);
        let desc: Vec<f64> = slots.iter().map(|s| s.0).collect();
        let Some(costs) = self.table.curve(base_rows, base_rows + desc.len()) else {
            return Decision::Warmup(base_ratio);
        };
        let b = optimal_budget(n as f64 + mandatory, &desc, &costs);
        let mut retained = vec![1usize; n];
        for &(_, i, j) in slots.iter().take(b) {
            retained[i] = retained[i].max(j + 1);
        }
        Decision::Retained(retained)
    }
}

static STATE: Mutex<Option<CostModel>> = Mutex::new(None);

fn with_model<R>(f: impl FnOnce(&mut CostModel) -> R) -> R {
    f(STATE.lock().get_or_insert_with(CostModel::new))
}

/// 3.7.4: feed a batched round of `rows` verify rows that took `wall_us`.
/// `mtp_stats::account_step` already feeds every batched MTP round; a second
/// feeder should call this only for rounds that step does not see.
pub fn observe_verify(rows: usize, wall_us: u64) {
    if enabled() {
        with_model(|m| m.observe_verify(rows, wall_us));
    }
}

/// 3.7.4: grade one sequence's pass. `conf_top1` is the drafter's top-1
/// LOG-probability of each draft the pass verified (`pending_draft_conf`), and
/// `accepted` how many of them were accepted.
pub fn observe_outcome(conf_top1: &[f32], accepted: usize) {
    if enabled() {
        with_model(|m| m.est.observe_pass(conf_top1, accepted));
    }
}

/// 3.7.4: see [`CostModel::budget`]. `None` when the row is off or uncalibrated.
pub fn budget(survival_desc: &[f32], non_draft_rows: usize) -> Option<usize> {
    if !enabled() {
        return None;
    }
    with_model(|m| m.budget(survival_desc, non_draft_rows))
}

/// The planner's decision for a verify batch (called by `mtp_dcut::plan`).
pub(in crate::scheduler) fn decide(
    confs: &[&[f32]],
    k: usize,
    row_budget: usize,
    base_ratio: f32,
) -> Decision {
    if !enabled() || k < 2 || confs.is_empty() {
        return Decision::Legacy;
    }
    with_model(|m| m.decide(confs, k, row_budget, base_ratio))
}

static PLANS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static ROWS_SUM: AtomicU64 = AtomicU64::new(0);
static ROWS_COUNT: AtomicU64 = AtomicU64::new(0);

/// Count one plan: how it was decided and how many verify rows it kept.
pub(in crate::scheduler) fn note_plan(d: &Decision, rows_kept: usize) {
    PLANS[d.tag()].fetch_add(1, Ordering::Relaxed);
    ROWS_SUM.fetch_add(rows_kept as u64, Ordering::Relaxed);
    ROWS_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// What `plan` leaves on a sequence for `account_step` to read after the step:
/// the verified depth and (row on) the retained drafts' confidences.
#[derive(Debug, Default, Clone)]
pub struct SeqDcut {
    pub planned: bool,
    pub verified: u32,
    pub conf: Vec<f32>,
}

/// Record a planned sequence's verified depth (call AFTER truncation).
pub(in crate::scheduler) fn stash(a: &mut ActiveSeq) {
    let verified = a.pending_drafts.len();
    let d = &mut a.port.dcut;
    d.planned = true;
    d.verified = u32::try_from(verified).unwrap_or(u32::MAX);
    d.conf.clear();
    if enabled() && a.pending_draft_conf.len() == verified {
        d.conf.extend_from_slice(&a.pending_draft_conf);
    }
}

/// Gathers one MTP step's per-sequence verdicts for [`StepFeed::finish`].
#[derive(Default)]
pub(in crate::scheduler) struct StepFeed {
    verifying: usize,
    planned: usize,
    bootstrap: usize,
    rows: usize,
    outcomes: Vec<(Vec<f32>, usize)>,
}

impl StepFeed {
    /// A sequence that only decoded and proposed (one row).
    pub fn bootstrap(&mut self) {
        self.bootstrap += 1;
        self.rows += 1;
    }

    /// A sequence that verified `proposed` pending drafts (fewer after a plan)
    /// and accepted `accepted` of them. Returns the depth it verified.
    pub fn verified(&mut self, d: SeqDcut, proposed: usize, accepted: usize) -> usize {
        let depth = if d.planned {
            (d.verified as usize).min(proposed)
        } else {
            proposed
        };
        self.verifying += 1;
        self.rows += depth + 1;
        if d.planned {
            self.planned += 1;
            if depth > 0 && d.conf.len() == depth {
                self.outcomes.push((d.conf, accepted.min(depth)));
            }
        }
        depth
    }

    /// Verify rows plus bootstrap rows of the step (the decode-rows gauge).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Feed the table and the estimator (row on only).
    pub fn finish(self, wall_us: u64) {
        if !enabled() || !super::dcut_enabled() {
            return;
        }
        with_model(|m| {
            for (conf, accepted) in &self.outcomes {
                m.est.observe_pass(conf, *accepted);
            }
            let one_batch = self.bootstrap == 0
                && self.verifying >= 2
                && self.planned == self.verifying
                && self.verifying <= WIDTH_CAP
                && self.rows <= VERIFY_ROW_BUDGET;
            if one_batch && (self.rows, wall_us) != m.last_fed {
                m.last_fed = (self.rows, wall_us);
                m.table.observe(self.rows, wall_us as f64);
                m.est.tick();
            }
        });
    }
}

/// `/metrics`: plans by source, the rows kept, and the calibration.
pub(in crate::scheduler) fn render_prometheus(out: &mut String) {
    let src = ["off", "warmup", "probe", "cost"];
    let labels: Vec<String> = src.iter().map(|s| format!("source=\"{s}\"")).collect();
    let plans: Vec<(&str, u64)> = labels
        .iter()
        .zip(&PLANS)
        .map(|(l, c)| (l.as_str(), c.load(Ordering::Relaxed)))
        .collect();
    pm::labeled_counter(
        out,
        "atlas_mtp_dcut_plans_total",
        "D-Cut verify-depth plans by how they were decided",
        &plans,
    );
    pm::hist_sum_count(
        out,
        "atlas_mtp_dcut_budget_rows",
        "Verify rows kept per D-Cut plan (sum and count)",
        ROWS_SUM.load(Ordering::Relaxed),
        ROWS_COUNT.load(Ordering::Relaxed),
    );
    let est = STATE
        .lock()
        .as_ref()
        .map_or_else(AcceptEstimator::new, |m| m.est.clone());
    let mut coefs: Vec<(String, f64)> = vec![("slope".into(), est.slope)];
    coefs.extend((0..MAX_POS.min(8)).map(|k| (format!("bias_{}", k + 1), est.bias[k])));
    coefs.push(("refits".into(), est.refits as f64));
    coefs.push(("observed".into(), est.observed as f64));
    let labels: Vec<String> = coefs.iter().map(|(c, _)| format!("coef=\"{c}\"")).collect();
    let series: Vec<(&str, f64)> = labels
        .iter()
        .zip(&coefs)
        .map(|(l, (_, v))| (l.as_str(), *v))
        .collect();
    pm::labeled(
        out,
        "atlas_mtp_dcut_calibration",
        "D-Cut v2 acceptance calibration: shared slope, per-position intercepts, refits",
        &series,
    );
}

#[cfg(test)]
#[path = "mtp_dcut_cost_tests.rs"]
mod tests;
