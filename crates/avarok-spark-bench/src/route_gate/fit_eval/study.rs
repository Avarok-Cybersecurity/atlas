// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Evaluation design after 80-routing.md section 5 item 4 (the B0 gate) and section 8.
//! Responsibility: the whole study on in-memory data: the arms, their AUROCs and intervals, calibration, and the gate
//! (AUROC >= 0.70 and >= the length/keyword baseline + 0.05, with the verdict and caveats).

use std::collections::BTreeMap;

use serde_json::Value;

use super::arms::{cv_best_candidate, neutral_fill, source_prior_scores, within_source_auroc};
use super::features::text_features;
use super::{
    BASELINE_ARM, GATE_ARM, GATE_AUROC, GATE_MARGIN, LOCAL_PROXY, Record, SMALL_PROXY, Stage1Input,
};
use crate::route_gate::error::GResult;
use crate::route_gate::mat::Mat;
use crate::route_gate::metrics::{
    Bin, BootDiff, auroc, bootstrap_auroc, brier, calibration_bins, cv_logistic, cv_platt, ece,
};

/// Knobs of [`run_study`].
#[derive(Clone, Debug)]
pub struct StudyParams {
    pub folds: usize,
    pub repeats: usize,
    pub seed: u64,
    pub bootstrap: usize,
    pub grid: Vec<f64>,
}

impl Default for StudyParams {
    fn default() -> Self {
        Self {
            folds: 5,
            repeats: 3,
            seed: 20_261_112,
            bootstrap: 1000,
            grid: super::DEFAULT_GRID.to_vec(),
        }
    }
}

/// Calibration of one arm.
#[derive(Clone, Debug)]
pub struct Calibration {
    pub bins: Vec<Bin>,
    pub ece: f64,
    pub brier: f64,
    pub calibrated_by: String,
}

/// One arm of the study.
#[derive(Clone, Debug)]
pub struct Arm {
    pub name: String,
    /// Out-of-fold score, higher means more likely to fail.
    pub score: Vec<f64>,
    /// Native probability, when the arm has one.
    pub prob: Option<Vec<f64>>,
    pub aurocs: Vec<f64>,
    pub kind: &'static str,
    pub desc: String,
    pub l2_chosen: Option<Vec<f64>>,
    pub picked: Option<Vec<(String, usize)>>,
    pub auroc: f64,
    pub auroc_of_mean_oof: f64,
    pub ci95: [f64; 2],
    pub flipped_auroc_diagnostic: f64,
    pub within_source: BTreeMap<String, f64>,
    pub diff_vs_baseline: Option<BootDiff>,
    pub calibration: Calibration,
}

/// The gate and its verdict.
#[derive(Clone, Debug)]
pub struct Gate {
    pub arm: String,
    pub baseline_arm: String,
    pub auroc: f64,
    pub baseline_auroc: f64,
    pub threshold: f64,
    pub margin: f64,
    pub point_pass: bool,
    pub strict_pass: bool,
    pub verdict: String,
    pub ci95: [f64; 2],
    pub diff_vs_baseline: Option<BootDiff>,
    pub within_source_mean: f64,
    pub oracle_pass: Option<bool>,
    pub caveats: Vec<String>,
    pub releases: String,
    pub if_hold: String,
}

/// Rows dropped before the study, by reason.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dropped {
    pub no_label: usize,
    pub no_readout: usize,
    pub truncated: usize,
    pub no_answer: usize,
}

/// Everything the study produced.
#[derive(Clone, Debug)]
pub struct Study {
    pub n: usize,
    pub prevalence_fail: f64,
    pub params: StudyParams,
    pub arms: Vec<Arm>,
    pub gate: Gate,
    pub notes: Vec<String>,
    pub y: Vec<bool>,
    pub ids: Vec<String>,
    pub sources: Vec<String>,
    pub counts_by_source: BTreeMap<String, usize>,
    pub dropped: Option<Dropped>,
    pub readout_meta: Value,
    pub regime: Option<String>,
}

impl Study {
    /// The arm called `name`.
    pub fn arm(&self, name: &str) -> Option<&Arm> {
        self.arms.iter().find(|a| a.name == name)
    }
}

fn nanmean_rows(p: &Mat, cols: &[usize]) -> Vec<f64> {
    (0..p.rows)
        .map(|i| {
            let vals: Vec<f64> = cols
                .iter()
                .map(|&c| p.at(i, c))
                .filter(|v| !v.is_nan())
                .collect();
            if vals.is_empty() {
                f64::NAN
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            }
        })
        .collect()
}

/// Adds an arm that is not fitted (a raw Stage-1 score): NaN scores become the median, which is noted.
fn push_raw(
    arms: &mut Vec<Arm>,
    notes: &mut Vec<String>,
    y: &[bool],
    name: &str,
    score: &[f64],
    desc: &str,
) -> GResult<()> {
    let (score, filled) = neutral_fill(score);
    if filled > 0 {
        notes.push(format!(
            "{name}: {filled} rows had no finite probability and were scored neutral"
        ));
    }
    let a = auroc(y, &score)?;
    arms.push(draft(name, score, None, vec![a], "raw", desc));
    Ok(())
}

fn draft(
    name: &str,
    score: Vec<f64>,
    prob: Option<Vec<f64>>,
    aurocs: Vec<f64>,
    kind: &'static str,
    desc: &str,
) -> Arm {
    Arm {
        name: name.to_string(),
        score,
        prob,
        aurocs,
        kind,
        desc: desc.to_string(),
        l2_chosen: None,
        picked: None,
        auroc: f64::NAN,
        auroc_of_mean_oof: f64::NAN,
        ci95: [f64::NAN, f64::NAN],
        flipped_auroc_diagnostic: f64::NAN,
        within_source: BTreeMap::new(),
        diff_vs_baseline: None,
        calibration: Calibration {
            bins: Vec::new(),
            ece: f64::NAN,
            brier: f64::NAN,
            calibrated_by: String::new(),
        },
    }
}

/// The whole study. `records` hold id, source, prompt and `passed`; `x` is `[n, d]` readouts; `stage1` optionally gives
/// the `[n, C]` point success probabilities for the raw ARC arms.
pub fn run_study(
    records: &[Record],
    x: &Mat,
    stage1: Option<&Stage1Input>,
    oracle_pass: Option<bool>,
    params: &StudyParams,
) -> GResult<Study> {
    let n = records.len();
    let y: Vec<bool> = records.iter().map(|r| !r.passed).collect();
    let sources: Vec<String> = records.iter().map(|r| r.source.clone()).collect();
    let (folds, repeats, seed) = (params.folds, params.repeats, params.seed);
    let mut arms: Vec<Arm> = Vec::new();
    let mut notes: Vec<String> = Vec::new();

    let add_logistic =
        |arms: &mut Vec<Arm>, name: &str, features: &Mat, desc: &str| -> GResult<()> {
            let res = cv_logistic(features, &y, &params.grid, folds, repeats, seed, 3)?;
            let mut arm = draft(
                name,
                res.oof.clone(),
                Some(res.oof),
                res.aurocs,
                "fitted",
                desc,
            );
            let mut l2 = res.l2;
            l2.sort_by(f64::total_cmp);
            l2.dedup();
            arm.l2_chosen = Some(l2);
            arms.push(arm);
            Ok(())
        };

    let rows: Vec<Vec<f64>> = records.iter().map(|r| text_features(&r.prompt)).collect();
    let feats = Mat::from_rows(&rows)?;
    add_logistic(
        &mut arms,
        BASELINE_ARM,
        &feats,
        "logistic on prompt length and keyword features",
    )?;
    let sp = source_prior_scores(&sources, &y, folds, seed);
    arms.push(draft(
        "source_prior",
        sp.clone(),
        Some(sp.clone()),
        vec![auroc(&y, &sp)?],
        "prior",
        "training failure rate of the prompt's dataset",
    ));
    add_logistic(
        &mut arms,
        GATE_ARM,
        x,
        "logistic on the 1024-d encoder readout",
    )?;

    if let Some(s1) = stage1 {
        let p = &s1.p;
        let names = &s1.names;
        let named: Vec<usize> = (0..names.len())
            .filter(|&i| !names[i].starts_with("withheld"))
            .collect();
        let all: Vec<usize> = (0..p.cols).collect();
        let inv = |v: Vec<f64>| v.into_iter().map(|a| 1.0 - a).collect::<Vec<f64>>();
        let desc_all = "1 - mean over finite candidates";
        push_raw(
            &mut arms,
            &mut notes,
            &y,
            "raw_mean_all",
            &inv(nanmean_rows(p, &all)),
            desc_all,
        )?;
        let desc_named = format!("1 - mean over {} named candidates", named.len());
        push_raw(
            &mut arms,
            &mut notes,
            &y,
            "raw_mean_named",
            &inv(nanmean_rows(p, &named)),
            &desc_named,
        )?;
        for (arm, who) in [
            ("raw_local_proxy", LOCAL_PROXY),
            ("raw_small_proxy", SMALL_PROXY),
        ] {
            if let Some(c) = names.iter().position(|nm| nm == who) {
                push_raw(
                    &mut arms,
                    &mut notes,
                    &y,
                    arm,
                    &inv(p.col(c)),
                    &format!("1 - p({who})"),
                )?;
            } else {
                notes.push(format!("{arm}: candidate {who} not in this release"));
            }
        }
        let best = cv_best_candidate(p, &y, folds, repeats, seed)?;
        let mut picked: BTreeMap<usize, usize> = BTreeMap::new();
        for &c in best.picks.iter().filter(|&&c| c >= 0) {
            *picked.entry(c as usize).or_insert(0) += 1;
        }
        let mut arm = draft(
            "raw_cv_best",
            best.oof,
            None,
            best.aurocs,
            "raw",
            "1 - p of the fold-selected best candidate",
        );
        arm.picked = Some(
            picked
                .into_iter()
                .map(|(c, k)| (names[c].clone(), k))
                .collect(),
        );
        arms.push(arm);
        let keep: Vec<usize> = (0..p.cols)
            .filter(|&c| {
                let col = p.col(c);
                let (lo, hi) = col
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                        (l.min(v), h.max(v))
                    });
                col.iter().all(|v| v.is_finite()) && hi - lo > 1e-12
            })
            .collect();
        if keep.is_empty() {
            notes.push(
                "logreg_stage1_p: no candidate column was finite and non-constant".to_string(),
            );
        } else {
            let sel = p.select_cols(&keep);
            add_logistic(
                &mut arms,
                "logreg_stage1_p",
                &sel,
                &format!("logistic on {} candidate probabilities", keep.len()),
            )?;
        }
    }

    let scores: Vec<(String, Vec<f64>)> = arms
        .iter()
        .map(|a| (a.name.clone(), a.score.clone()))
        .collect();
    let boot = bootstrap_auroc(&y, &scores, params.bootstrap, seed, Some(BASELINE_ARM))?;
    for a in &mut arms {
        a.auroc = a.aurocs.iter().sum::<f64>() / a.aurocs.len() as f64;
        a.auroc_of_mean_oof = auroc(&y, &a.score)?;
        let ci = boot.ci(&a.name).map_or([f64::NAN; 2], |c| [c.lo, c.hi]);
        a.ci95 = ci;
        a.flipped_auroc_diagnostic = 1.0 - a.auroc_of_mean_oof;
        a.within_source = within_source_auroc(&y, &a.score, &sources, 10)?;
        a.diff_vs_baseline = boot.diff_of(&a.name);
        let (prob, by) = match &a.prob {
            Some(p) => (p.clone(), "native probability"),
            None => (
                cv_platt(&a.score, &y, folds, 1, seed),
                "cv platt on the score",
            ),
        };
        let bins = calibration_bins(&y, &prob, 10);
        a.calibration = Calibration {
            ece: ece(&bins),
            brier: brier(&y, &prob),
            bins,
            calibrated_by: by.to_string(),
        };
    }
    let gate = build_gate(&arms, n, &y, oracle_pass, stage1.is_some());
    let mut counts_by_source = BTreeMap::new();
    for s in &sources {
        *counts_by_source.entry(s.clone()).or_insert(0usize) += 1;
    }
    let prevalence_fail = y.iter().filter(|&&v| v).count() as f64 / n as f64;
    Ok(Study {
        n,
        prevalence_fail,
        params: params.clone(),
        arms,
        gate,
        notes,
        y,
        ids: records.iter().map(|r| r.id.clone()).collect(),
        sources,
        counts_by_source,
        dropped: None,
        readout_meta: Value::Null,
        regime: None,
    })
}

fn build_gate(
    arms: &[Arm],
    n: usize,
    y: &[bool],
    oracle_pass: Option<bool>,
    had_stage1: bool,
) -> Gate {
    let find = |name: &str| arms.iter().find(|a| a.name == name).expect("arm present");
    let (gate_arm, base) = (find(GATE_ARM), find(BASELINE_ARM));
    let (a, b) = (gate_arm.auroc, base.auroc);
    let diff_lo = gate_arm.diff_vs_baseline.map_or(f64::NAN, |d| d.lo);
    let point_pass = a >= GATE_AUROC && a >= b + GATE_MARGIN;
    let strict_pass = gate_arm.ci95[0] >= GATE_AUROC && diff_lo > 0.0;
    let prevalence = y.iter().filter(|&&v| v).count() as f64 / y.len() as f64;
    let ws = &gate_arm.within_source;
    let ws_mean = if ws.is_empty() {
        f64::NAN
    } else {
        ws.values().sum::<f64>() / ws.len() as f64
    };
    let mut caveats = Vec::new();
    if !(0.10..=0.90).contains(&prevalence) {
        caveats.push(format!(
            "failure rate {prevalence:.2} is outside 0.10..0.90: the AUROC is noisy"
        ));
    }
    if n < 300 {
        caveats.push(format!("only {n} labeled prompts (plan: about 600)"));
    }
    let prior = find("source_prior").auroc;
    let confound = prior >= a - 0.03 || (!ws.is_empty() && ws_mean < 0.60);
    if confound {
        caveats.push(format!(
            "source confound: source_prior AUROC {prior:.3}, gate arm within-source mean {ws_mean:.3}; the readout may only identify the dataset"
        ));
    }
    if !strict_pass {
        caveats.push(
            "the 95% bootstrap interval does not clear the gate (AUROC lower bound < 0.70 or the paired difference to the baseline includes 0)"
                .to_string(),
        );
    }
    caveats.push(
        "labels are one greedy sample per prompt (thinking off): label noise lowers every AUROC"
            .to_string(),
    );
    if !had_stage1 {
        caveats.push("Stage-1 raw arms were not run (no --release)".to_string());
    }
    let verdict = if oracle_pass != Some(true) {
        caveats.insert(
            0,
            "the signal-head oracle (route_gate goldens) has not passed or its report was not given: the readouts are not shown to match the reference encoder, so no AUROC may be believed yet".to_string(),
        );
        "UNVERIFIED"
    } else if point_pass && !confound {
        "RELEASE"
    } else if point_pass {
        "RELEASE_WITH_CAVEATS"
    } else {
        "HOLD"
    };
    let releases = if matches!(verdict, "RELEASE" | "RELEASE_WITH_CAVEATS") {
        "X01, X02, X03 (endpoint), X05, X06"
    } else {
        "none"
    };
    Gate {
        arm: GATE_ARM.to_string(),
        baseline_arm: BASELINE_ARM.to_string(),
        auroc: a,
        baseline_auroc: b,
        threshold: GATE_AUROC,
        margin: GATE_MARGIN,
        point_pass,
        strict_pass,
        verdict: verdict.to_string(),
        ci95: gate_arm.ci95,
        diff_vs_baseline: gate_arm.diff_vs_baseline,
        within_source_mean: ws_mean,
        oracle_pass,
        caveats,
        releases: releases.to_string(),
        if_hold: "use the B1 confidence summary plus heuristics (L05, L03 route_bench) instead"
            .to_string(),
    }
}
