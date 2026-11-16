// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Evaluation design written for Atlas (the B0 gate of the routing study); no upstream code.
//! Responsibility: step 4 of the routing gate (B0): does a routing signal predict "the 4B fails this prompt"?
//!
//! Target: `y = true` when the 4B FAILED the prompt. Every score is oriented "higher = more likely to fail". The
//! orientation of the raw ARC probabilities (`score = 1 - p`) is fixed before any data is seen; the opposite
//! orientation is printed only as a diagnostic and never used for the gate.
//!
//! Arms (all out-of-fold; repeated stratified 5-fold; ridge strength chosen inside each training fold):
//!
//! | arm | what |
//! |---|---|
//! | `len_kw` | logistic head on prompt length and keyword features only (the baseline the gate must beat) |
//! | `source_prior` | the training failure rate of the prompt's dataset (shows how much is just "which dataset") |
//! | `logreg_readout` | logistic head on the 1024-d encoder readout (THE GATE ARM) |
//! | `raw_mean_all` | 1 - mean over every finite candidate p (no fitting; Stage-1 head, regime fixed) |
//! | `raw_mean_named` | 1 - mean over the 11 named candidates (no fitting) |
//! | `raw_local_proxy` | 1 - p(deepseek-v4.1-flash), the model the release's prefer-local evaluation runs locally |
//! | `raw_small_proxy` | 1 - p(claude-haiku-5.5), the smallest named model |
//! | `raw_cv_best` | 1 - p of the single candidate with the best TRAIN AUROC in each fold (nested selection) |
//! | `logreg_stage1_p` | logistic head on the 56 candidate probabilities |
//!
//! The study design names `p(qwen3.8-27b)` as the raw arm, but the release has no such candidate (11 named models, 45
//! withheld), so the proxies above stand in; this is reported, not hidden.
//!
//! Gate: `AUROC(gate arm) >= 0.70` AND `>= AUROC(len_kw) + 0.05`. Nothing is believed until the
//! signal-head oracle (`route_gate goldens`) has passed; without its report the verdict is UNVERIFIED.

mod arms;
mod features;
mod inputs;
mod report;
mod study;

pub use arms::{
    BestCandidate, cv_best_candidate, neutral_fill, source_prior_scores, within_source_auroc,
};
pub use features::{KEYWORDS, feature_names, make_synthetic, text_features};
pub use inputs::{Inputs, load_inputs, write_readouts};
pub use report::{markdown_report, report_json, write_oof, write_outputs, write_outputs_with_oof};
pub use study::{Arm, Calibration, Dropped, Gate, Study, StudyParams, run_study};

use std::path::PathBuf;

use anyhow::{Context, Result};

use super::mat::Mat;
use super::stage_one::StageOne;

pub const GATE_ARM: &str = "logreg_readout";
pub const BASELINE_ARM: &str = "len_kw";
pub const GATE_AUROC: f64 = 0.70;
pub const GATE_MARGIN: f64 = 0.05;
pub const LOCAL_PROXY: &str = "deepseek-v4.1-flash";
pub const SMALL_PROXY: &str = "claude-haiku-5.5";
pub const DEFAULT_GRID: [f64; 6] = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5];

/// One labeled prompt of the study.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub id: String,
    pub source: String,
    pub prompt: String,
    pub passed: bool,
}

/// The Stage-1 point probabilities `[n, C]` and the candidate names, for the raw arms.
#[derive(Clone, Debug)]
pub struct Stage1Input {
    pub p: Mat,
    pub names: Vec<String>,
}

/// Arguments of `fit`.
pub struct FitArgs {
    pub labels: PathBuf,
    pub readouts: PathBuf,
    pub release: Option<PathBuf>,
    pub no_verify: bool,
    pub regime: String,
    pub oracle_report: Option<PathBuf>,
    pub out_dir: PathBuf,
    pub params: StudyParams,
    pub drop_truncated: bool,
    pub drop_no_answer: bool,
}

/// `fit`: the study on real labels and readouts. Exit code 0 on RELEASE or RELEASE_WITH_CAVEATS, 1 otherwise, 2 when
/// there are too few usable rows.
pub fn cmd_fit(a: &FitArgs) -> Result<i32> {
    let inputs = load_inputs(&a.labels, &a.readouts, a.drop_truncated, a.drop_no_answer)?;
    if inputs.rows.len() < 40 {
        eprintln!(
            "fit_eval: only {} usable rows (dropped {:?}); need at least 40",
            inputs.rows.len(),
            inputs.dropped
        );
        return Ok(2);
    }
    let stage1 = match &a.release {
        Some(dir) => {
            let head = StageOne::from_release(dir, !a.no_verify)?;
            let p = head.point_p_one(&inputs.x, &a.regime)?;
            Some(Stage1Input {
                p,
                names: head.candidate_names(),
            })
        }
        None => None,
    };
    let oracle = match &a.oracle_report {
        Some(path) => {
            let v: serde_json::Value = serde_json::from_slice(
                &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
            )?;
            Some(v["oracle_pass"].as_bool().unwrap_or(false))
        }
        None => None,
    };
    let mut study = run_study(&inputs.rows, &inputs.x, stage1.as_ref(), oracle, &a.params)?;
    study.dropped = Some(inputs.dropped.clone());
    study.readout_meta = inputs.meta.clone();
    study.regime = Some(a.regime.clone());
    write_outputs_with_oof(&study, &a.out_dir)?;
    println!("{}", markdown_report(&study));
    Ok(i32::from(!matches!(
        study.gate.verdict.as_str(),
        "RELEASE" | "RELEASE_WITH_CAVEATS"
    )))
}

/// `fit --selftest`: the study on planted synthetic data.
pub fn cmd_selftest() -> Result<i32> {
    let (recs, x) = make_synthetic(240, 24, 0, 3.0);
    let params = StudyParams {
        folds: 3,
        repeats: 1,
        bootstrap: 100,
        grid: vec![10.0, 100.0],
        ..StudyParams::default()
    };
    let res = run_study(&recs, &x, None, Some(true), &params)?;
    let g = &res.gate;
    println!(
        "fit_eval selftest: gate arm {:.3}, baseline {:.3}, verdict {}",
        g.auroc, g.baseline_auroc, g.verdict
    );
    let ok = g.auroc > 0.65 && g.auroc > g.baseline_auroc;
    println!("fit_eval selftest: {}", if ok { "PASS" } else { "FAIL" });
    Ok(i32::from(!ok))
}
