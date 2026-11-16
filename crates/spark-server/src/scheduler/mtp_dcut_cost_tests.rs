// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (`mtp_dcut_cost`). A sibling file via `#[path]`, like
//! `mtp_dcut_tests.rs`, so the module stays under the 500-line cap.

use super::super::select;
use super::*;

/// Calibrate `rows` with `MIN_SAMPLES` identical measurements.
fn fill(m: &mut CostModel, rows: usize, cost: f64) {
    for _ in 0..MIN_SAMPLES {
        m.table.observe(rows, cost);
    }
}

fn refs(c: &[Vec<f32>]) -> Vec<&[f32]> {
    c.iter().map(|v| v.as_slice()).collect()
}

/// Four sequences, three drafts each, with well-separated prefix scores.
fn confs4() -> Vec<Vec<f32>> {
    vec![
        vec![-0.05, -0.10, -0.30],
        vec![-0.30, -0.20, -0.60],
        vec![-0.10, -0.90, -0.15],
        vec![-0.60, -0.05, -1.50],
    ]
}

/// Deterministic xorshift stream of integers.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) % n
    }
}

#[test]
fn budget_argmax_equals_brute_force_over_every_b() {
    // Dyadic survivals and integer costs make every ratio exact, so the
    // brute-force maximum and the scan see identical f64 values.
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for _ in 0..500 {
        let len = rng.below(12) as usize;
        let mut surv: Vec<f64> = (0..len)
            .map(|_| (1 + rng.below(64)) as f64 / 64.0)
            .collect();
        surv.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut cost = 50.0 + rng.below(100) as f64;
        let costs: Vec<f64> = (0..=len)
            .map(|_| {
                cost += rng.below(300) as f64;
                cost
            })
            .collect();
        let base = (1 + rng.below(8)) as f64;
        let value = |b: usize| (base + surv[..b].iter().sum::<f64>()) / costs[b];
        let best = (0..=len).map(value).fold(f64::NEG_INFINITY, f64::max);
        let want = (0..=len).rev().find(|&b| value(b) == best).unwrap();
        assert_eq!(
            optimal_budget(base, &surv, &costs),
            want,
            "surv={surv:?} costs={costs:?}"
        );
    }
}

#[test]
fn a_flat_cost_table_selects_the_full_budget() {
    let mut m = CostModel::new();
    for r in 8..=16 {
        fill(&mut m, r, 100.0);
    }
    let c = confs4();
    match m.decide(&refs(&c), 3, 160, 0.75) {
        Decision::Retained(r) => assert_eq!(r, vec![3, 3, 3, 3]),
        other => panic!("expected the cost-optimal plan, got {other:?}"),
    }
}

#[test]
fn a_steep_cost_table_keeps_only_the_mandatory_draft() {
    let mut m = CostModel::new();
    for r in 8..=16usize {
        fill(&mut m, r, 100.0 * 4f64.powi((r - 8) as i32));
    }
    let c = confs4();
    match m.decide(&refs(&c), 3, 160, 0.75) {
        Decision::Retained(r) => assert_eq!(r, vec![1, 1, 1, 1]),
        other => panic!("expected the cost-optimal plan, got {other:?}"),
    }
}

#[test]
fn equal_budget_selection_equals_the_fixed_ratio_select() {
    // slope 1, intercept 0 makes the calibrated p equal the drafter's own q, so
    // the survival ranking is the log-probability prefix-sum ranking `select` uses.
    let mut m = CostModel::new();
    m.est.slope = 1.0;
    m.est.bias = [0.0; MAX_POS];
    // Rows up to 13 (b <= 5 optional drafts) are cheap, beyond that ruinous.
    for r in 8..=16 {
        fill(&mut m, r, if r <= 13 { 100.0 } else { 10_000.0 });
    }
    let c = confs4();
    let want = select(&refs(&c), 3, 160, 5.0 / 8.0);
    assert_eq!(want, vec![3, 2, 2, 2], "fixture sanity");
    match m.decide(&refs(&c), 3, 160, 0.75) {
        Decision::Retained(r) => assert_eq!(r, want),
        other => panic!("expected the cost-optimal plan, got {other:?}"),
    }
}

#[test]
fn the_row_budget_caps_the_optional_drafts() {
    let mut m = CostModel::new();
    for r in 8..=16 {
        fill(&mut m, r, 100.0);
    }
    let c = confs4();
    // 11 rows allowed: 8 committed, 3 spare.
    match m.decide(&refs(&c), 3, 11, 0.75) {
        Decision::Retained(r) => {
            let rows: usize = r.iter().map(|k| k + 1).sum();
            assert_eq!(rows, 11, "{r:?}");
            assert!(r.iter().all(|k| (1..=3).contains(k)));
        }
        other => panic!("expected the cost-optimal plan, got {other:?}"),
    }
}

#[test]
fn an_uncalibrated_table_falls_back_and_probes_full_width() {
    let mut m = CostModel::new();
    let c = confs4();
    let d: Vec<Decision> = (0..8).map(|_| m.decide(&refs(&c), 3, 160, 0.75)).collect();
    assert_eq!(d[0], Decision::Warmup(0.75));
    assert_eq!(d[2], Decision::Warmup(0.75));
    assert_eq!(d[3], Decision::Probe, "every 4th plan probes");
    assert_eq!(d[7], Decision::Probe);
    // Bottom-only data does not calibrate the top of the range.
    let mut m = CostModel::new();
    fill(&mut m, 8, 100.0);
    assert!(!m.table.covers(16));
    assert_eq!(m.decide(&refs(&c), 3, 160, 0.5), Decision::Warmup(0.5));
}

#[test]
fn a_calibrated_table_reprobes_full_width_periodically() {
    let mut m = CostModel::new();
    for r in 8..=16 {
        fill(&mut m, r, 100.0);
    }
    let c = confs4();
    m.plans = REPROBE_PERIOD - 1;
    assert_eq!(m.decide(&refs(&c), 3, 160, 0.75), Decision::Probe);
    assert!(matches!(
        m.decide(&refs(&c), 3, 160, 0.75),
        Decision::Retained(_)
    ));
}

#[test]
fn an_entry_counts_only_after_min_samples() {
    let mut t = CostTable::new();
    for _ in 0..MIN_SAMPLES - 1 {
        t.observe(12, 50.0);
    }
    assert!(!t.covers(12));
    assert!(t.curve(12, 12).is_none());
    t.observe(12, 50.0);
    assert!(t.covers(12));
    assert_eq!(t.curve(12, 12), Some(vec![50.0]));
}

#[test]
fn the_curve_interpolates_clamps_below_and_refuses_above() {
    let mut t = CostTable::new();
    for _ in 0..MIN_SAMPLES {
        t.observe(8, 100.0);
        t.observe(16, 200.0);
    }
    let c = t.curve(6, 16).expect("covered");
    assert_eq!(c.len(), 11);
    assert_eq!(
        (c[0], c[1], c[2]),
        (100.0, 100.0, 100.0),
        "below the lowest point"
    );
    assert!(
        (c[6] - 150.0).abs() < 1e-9,
        "rows 12 is the midpoint: {}",
        c[6]
    );
    assert_eq!(c[10], 200.0);
    assert!(c.windows(2).all(|w| w[0] <= w[1]));
    assert!(
        t.curve(8, 17).is_none(),
        "above the highest calibrated point"
    );
}

#[test]
fn a_noisy_dip_is_flattened_by_the_running_max() {
    let mut t = CostTable::new();
    for _ in 0..MIN_SAMPLES {
        t.observe(8, 100.0);
        t.observe(12, 90.0);
    }
    assert_eq!(t.curve(12, 12), Some(vec![100.0]));
}

#[test]
fn the_ewma_follows_a_shift_in_cost() {
    let mut t = CostTable::new();
    for _ in 0..MIN_SAMPLES {
        t.observe(10, 100.0);
    }
    for _ in 0..40 {
        t.observe(10, 200.0);
    }
    let v = t.curve(10, 10).unwrap()[0];
    assert!(v > 190.0 && v <= 200.0, "{v}");
    // Garbage is ignored.
    t.observe(10, f64::NAN);
    t.observe(10, -5.0);
    t.observe(0, 5.0);
    t.observe(VERIFY_ROW_BUDGET + 1, 5.0);
    assert_eq!(t.curve(10, 10).unwrap()[0], v);
}

#[test]
fn the_public_budget_needs_a_calibrated_range() {
    let mut m = CostModel::new();
    let surv = [0.9f32, 0.5, 0.2];
    assert_eq!(m.budget(&surv, 4), None);
    for r in 4..=7 {
        fill(&mut m, r, 100.0);
    }
    assert_eq!(m.budget(&surv, 4), Some(3), "flat cost keeps everything");
    // Input order does not matter: the survivals are ranked first.
    assert_eq!(m.budget(&[0.2, 0.9, 0.5], 4), Some(3));
    // A cost cliff after the first optional row keeps one.
    let mut m = CostModel::new();
    fill(&mut m, 4, 100.0);
    fill(&mut m, 5, 101.0);
    fill(&mut m, 6, 5_000.0);
    fill(&mut m, 7, 9_000.0);
    assert_eq!(m.budget(&surv, 4), Some(1));
}

#[test]
fn a_duplicate_feed_of_one_round_counts_once() {
    let mut m = CostModel::new();
    m.observe_verify(12, 1_000);
    m.observe_verify(12, 1_000);
    assert_eq!(m.table.n[12], 1);
    m.observe_verify(12, 1_100);
    assert_eq!(m.table.n[12], 2);
}

#[test]
fn step_feed_counts_rows_and_keeps_the_verified_depth() {
    let mut f = StepFeed::default();
    let planned = SeqDcut {
        planned: true,
        verified: 2,
        conf: vec![-0.1, -0.2],
    };
    // The sequence proposed 3 drafts but the plan verified 2 of them.
    assert_eq!(f.verified(planned, 3, 1), 2);
    // An unplanned (serial) sequence verified everything it proposed.
    assert_eq!(f.verified(SeqDcut::default(), 3, 3), 3);
    f.bootstrap();
    assert_eq!(f.rows(), (2 + 1) + (3 + 1) + 1);
    assert_eq!(f.outcomes.len(), 1);
    assert_eq!(f.outcomes[0].1, 1);
    // Row off in the test environment: finishing is a no-op, not a panic.
    f.finish(1_000);
}

#[test]
fn the_row_is_off_by_default_so_the_planner_is_d_cut_v1() {
    if enabled() {
        return; // the environment turned the row on
    }
    let c = confs4();
    assert_eq!(decide(&refs(&c), 3, 160, 0.75), Decision::Legacy);
    assert_eq!(budget(&[0.5], 4), None);
    observe_verify(8, 100);
    observe_outcome(&[-0.1], 1);
    assert!(
        STATE.lock().is_none(),
        "nothing may allocate state while the row is off"
    );
}

#[test]
fn metrics_render_the_planner_series() {
    note_plan(&Decision::Legacy, 24);
    let mut s = String::new();
    render_prometheus(&mut s);
    for needle in [
        "# TYPE atlas_mtp_dcut_plans_total counter",
        "atlas_mtp_dcut_plans_total{source=\"off\"}",
        "atlas_mtp_dcut_plans_total{source=\"cost\"}",
        "# TYPE atlas_mtp_dcut_budget_rows summary",
        "atlas_mtp_dcut_budget_rows_sum ",
        "atlas_mtp_dcut_budget_rows_count ",
        "atlas_mtp_dcut_calibration{coef=\"slope\"}",
        "atlas_mtp_dcut_calibration{coef=\"bias_1\"}",
        "atlas_mtp_dcut_calibration{coef=\"refits\"}",
    ] {
        assert!(s.contains(needle), "missing {needle:?} in\n{s}");
    }
}
