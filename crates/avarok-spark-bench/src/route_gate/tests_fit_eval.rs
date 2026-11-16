// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: unit tests of `fit_eval` (features, source prior, nested best candidate, the study on planted and
//! noise data, raw arms and report files, input join), ported from the Python `FitEvalTests`.

use std::collections::BTreeSet;

use serde_json::{Value, json};

use super::fit_eval::{
    Dropped, Record, Stage1Input, StudyParams, cv_best_candidate, feature_names, load_inputs,
    make_synthetic, run_study, source_prior_scores, text_features, write_outputs_with_oof,
    write_readouts,
};
use super::mat::Mat;
use super::metrics::auroc;
use super::mt::NumpyRandom;

fn params(folds: usize, repeats: usize, bootstrap: usize, grid: &[f64]) -> StudyParams {
    StudyParams {
        folds,
        repeats,
        bootstrap,
        grid: grid.to_vec(),
        ..StudyParams::default()
    }
}

#[test]
fn text_features_shape() {
    let f = text_features("Prove that the sum of two odd integers is even. How many ways?");
    let names = feature_names();
    assert_eq!(f.len(), names.len());
    assert!(f.iter().all(|v| v.is_finite()));
    let at = names.iter().position(|n| n == "kw:prove").unwrap();
    assert_eq!(f[at], 1.0);
}

#[test]
fn source_prior_uses_training_rows_only() {
    let sources: Vec<String> = (0..40)
        .map(|i| if i < 20 { "a" } else { "b" }.to_string())
        .collect();
    let mut y = vec![true; 18];
    y.extend([false; 2]);
    y.extend([false; 18]);
    y.extend([true; 2]);
    let sp = source_prior_scores(&sources, &y, 4, 0);
    assert!(sp.iter().all(|v| v.is_finite()));
    assert!(auroc(&y, &sp).unwrap() > 0.8);
}

#[test]
fn cv_best_candidate_finds_the_planted_column() {
    let mut rng = NumpyRandom::new(0);
    let (n, cn) = (200usize, 6usize);
    let y: Vec<bool> = rng.rand_vec(n).into_iter().map(|v| v < 0.4).collect();
    let mut p = rng.rand_vec(n * cn);
    let noise = rng.randn_vec(n);
    for i in 0..n {
        // lower success probability when the 4B fails
        p[i * cn + 2] = (0.7 - 0.5 * f64::from(u8::from(y[i])) + noise[i] * 0.1).clamp(0.01, 0.99);
    }
    let pm = Mat::new(n, cn, p).unwrap();
    let res = cv_best_candidate(&pm, &y, 4, 2, 0).unwrap();
    assert!(res.aurocs[0] > 0.9, "auroc {}", res.aurocs[0]);
    let hits = res.picks.iter().filter(|&&c| c == 2).count();
    assert!(hits >= res.picks.len() - 1, "{:?}", res.picks);
}

#[test]
fn study_on_planted_signal() {
    let (recs, x) = make_synthetic(240, 24, 0, 3.0);
    let p = params(3, 1, 60, &[10.0, 100.0]);
    let res = run_study(&recs, &x, None, Some(true), &p).unwrap();
    let g = &res.gate;
    assert!(g.auroc > 0.7, "gate auroc {}", g.auroc);
    assert!(g.auroc > res.arm("len_kw").unwrap().auroc);
    assert!(
        matches!(g.verdict.as_str(), "RELEASE" | "RELEASE_WITH_CAVEATS"),
        "{}",
        g.verdict
    );
    let names: BTreeSet<&str> = res.arms.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(
        names,
        BTreeSet::from(["len_kw", "source_prior", "logreg_readout"])
    );
    let unverified = run_study(&recs, &x, None, None, &params(3, 1, 30, &[10.0])).unwrap();
    assert_eq!(unverified.gate.verdict, "UNVERIFIED");
    assert_eq!(unverified.gate.releases, "none");
}

#[test]
fn study_on_noise_holds() {
    let mut rng = NumpyRandom::new(1);
    let mut recs = Vec::new();
    for i in 0..200 {
        let passed = rng.rand() < 0.5;
        let reps = rng.randint(3, 20) as usize;
        recs.push(Record {
            id: format!("n{i}"),
            source: "gsm8k".to_string(),
            passed,
            prompt: format!("Solve {}", "x ".repeat(reps)),
        });
    }
    let x = Mat::new(200, 16, rng.randn_vec(200 * 16)).unwrap();
    let res = run_study(
        &recs,
        &x,
        None,
        Some(true),
        &params(3, 1, 30, &[10.0, 100.0]),
    )
    .unwrap();
    assert_eq!(res.gate.verdict, "HOLD");
    assert!(!res.gate.point_pass);
}

#[test]
fn raw_arms_and_outputs() {
    let (recs, x) = make_synthetic(160, 12, 2, 3.0);
    let mut rng = NumpyRandom::new(3);
    let y: Vec<bool> = recs.iter().map(|r| !r.passed).collect();
    let (n, cn) = (recs.len(), 6usize);
    let mut p = rng.rand_vec(n * cn);
    let noise = rng.randn_vec(n);
    for i in 0..n {
        p[i * cn + 4] = (0.8 - 0.5 * f64::from(u8::from(y[i])) + noise[i] * 0.1).clamp(0.01, 0.99);
    }
    for i in 0..5 {
        // a candidate that is unscored for a few rows is skipped by the column-wise arms
        p[i * cn + 1] = f64::NAN;
    }
    let names: Vec<String> = [
        "withheld-model-00",
        "claude-haiku-5.5",
        "withheld-model-01",
        "withheld-model-02",
        "deepseek-v4.1-flash",
        "glm-5.3",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let s1 = Stage1Input {
        p: Mat::new(n, cn, p).unwrap(),
        names,
    };
    let sp = StudyParams {
        folds: 3,
        repeats: 1,
        seed: 7,
        bootstrap: 40,
        grid: vec![10.0],
    };
    let res = run_study(&recs, &x, Some(&s1), Some(true), &sp).unwrap();
    for arm in [
        "raw_mean_all",
        "raw_mean_named",
        "raw_local_proxy",
        "raw_small_proxy",
        "raw_cv_best",
        "logreg_stage1_p",
    ] {
        assert!(res.arm(arm).is_some(), "missing arm {arm}");
    }
    let local = res.arm("raw_local_proxy").unwrap();
    assert!(local.auroc > 0.8, "{}", local.auroc);
    assert!(
        local.flipped_auroc_diagnostic < 0.2,
        "{}",
        local.flipped_auroc_diagnostic
    );
    let tmp = tempfile::tempdir().unwrap();
    write_outputs_with_oof(&res, tmp.path()).unwrap();
    for name in [
        "report.json",
        "report.md",
        "calibration.csv",
        "oof_scores.csv",
    ] {
        assert!(
            std::fs::metadata(tmp.path().join(name)).unwrap().len() > 0,
            "{name}"
        );
    }
    let rep: Value =
        serde_json::from_str(&std::fs::read_to_string(tmp.path().join("report.json")).unwrap())
            .unwrap();
    assert!(rep.get("gate").is_some());
    assert!(rep["arms"]["len_kw"].get("score").is_none());
    assert!(
        std::fs::read_to_string(tmp.path().join("report.md"))
            .unwrap()
            .contains("Verdict")
    );
}

#[test]
fn load_inputs_joins_and_drops() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path();
    let ids: Vec<String> = (0..6).map(|i| format!("p{i}")).collect();
    let label = |id: &str, passed: Value, prompt: &str, extra: Value| {
        let mut r = json!({"id": id, "passed": passed, "source": "s", "prompt": prompt});
        if let Value::Object(e) = extra {
            for (k, v) in e {
                r[k] = v;
            }
        }
        r
    };
    let labels = [
        label("p0", json!(true), "a", json!({})),
        label("p1", json!(false), "b", json!({})),
        label("p2", Value::Null, "c", json!({})),
        label("p3", json!(false), "d", json!({"truncated": true})),
        label("p4", json!(false), "e", json!({"reason": "no_answer"})),
        label("p5", json!(true), "f", json!({})),
    ];
    let text: String = labels
        .iter()
        .map(|r| format!("{}\n", serde_json::to_string(r).unwrap()))
        .collect();
    std::fs::write(d.join("l.jsonl"), text).unwrap();
    let x: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let ok = [true, true, true, true, true, false];
    write_readouts(
        &d.join("r.safetensors"),
        &ids,
        &x,
        2,
        &[0, 1, 2, 3, 4, 5],
        &ok,
        &json!({"k": 1}),
    )
    .unwrap();
    let inp = load_inputs(&d.join("l.jsonl"), &d.join("r.safetensors"), true, true).unwrap();
    assert_eq!(
        inp.rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["p0", "p1"]
    );
    assert_eq!((inp.x.rows, inp.x.cols), (2, 2));
    assert_eq!(inp.x.data, vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(
        inp.dropped,
        Dropped {
            no_label: 1,
            no_readout: 1,
            truncated: 1,
            no_answer: 1
        }
    );
    assert_eq!(inp.meta, json!({"k": 1}));
    let inp = load_inputs(&d.join("l.jsonl"), &d.join("r.safetensors"), false, false).unwrap();
    assert_eq!(inp.rows.len(), 4);
}
