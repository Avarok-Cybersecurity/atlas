// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Report writers of the routing gate; no upstream code.
//! Responsibility: `report.md` (the table of AUROCs with intervals, caveats, verdict), `report.json` (everything but
//! the per-prompt scores), `calibration.csv` and `oof_scores.csv`.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;
use serde_json::{Map, Value, json};

use super::GATE_ARM;
use super::study::{Arm, Gate, Study};

fn num(v: f64) -> Value {
    json!(v)
}

fn diff_json(d: Option<crate::route_gate::metrics::BootDiff>) -> Value {
    d.map_or(
        Value::Null,
        |d| json!({"mean": d.mean, "lo": d.lo, "hi": d.hi}),
    )
}

fn arm_json(a: &Arm) -> Value {
    let mut m = Map::new();
    m.insert("aurocs".into(), json!(a.aurocs));
    m.insert("kind".into(), json!(a.kind));
    m.insert("desc".into(), json!(a.desc));
    if let Some(l2) = &a.l2_chosen {
        m.insert("l2_chosen".into(), json!(l2));
    }
    if let Some(p) = &a.picked {
        m.insert(
            "picked".into(),
            Value::Object(p.iter().map(|(k, v)| (k.clone(), json!(v))).collect()),
        );
    }
    m.insert("auroc".into(), num(a.auroc));
    m.insert("auroc_of_mean_oof".into(), num(a.auroc_of_mean_oof));
    m.insert("ci95".into(), json!([a.ci95[0], a.ci95[1]]));
    m.insert(
        "flipped_auroc_diagnostic".into(),
        num(a.flipped_auroc_diagnostic),
    );
    m.insert(
        "within_source".into(),
        Value::Object(
            a.within_source
                .iter()
                .map(|(k, v)| (k.clone(), num(*v)))
                .collect(),
        ),
    );
    m.insert("diff_vs_baseline".into(), diff_json(a.diff_vs_baseline));
    let bins: Vec<Value> = a
        .calibration
        .bins
        .iter()
        .map(|b| json!({"n": b.n, "mean_pred": b.mean_pred, "frac_pos": b.frac_pos}))
        .collect();
    m.insert(
        "calibration".into(),
        json!({"bins": bins, "ece": a.calibration.ece, "brier": a.calibration.brier, "calibrated_by": a.calibration.calibrated_by}),
    );
    Value::Object(m)
}

fn gate_json(g: &Gate) -> Value {
    json!({
        "arm": g.arm, "baseline_arm": g.baseline_arm, "auroc": g.auroc, "baseline_auroc": g.baseline_auroc,
        "threshold": g.threshold, "margin": g.margin, "point_pass": g.point_pass, "strict_pass": g.strict_pass,
        "verdict": g.verdict, "ci95": [g.ci95[0], g.ci95[1]], "diff_vs_baseline": diff_json(g.diff_vs_baseline),
        "within_source_mean": g.within_source_mean, "oracle_pass": g.oracle_pass, "caveats": g.caveats,
        "releases": g.releases, "if_hold": g.if_hold,
    })
}

/// The `report.json` document: the study without the per-prompt scores. Non-finite numbers become `null`.
pub fn report_json(s: &Study) -> Value {
    let mut m = Map::new();
    m.insert("n".into(), json!(s.n));
    m.insert("prevalence_fail".into(), num(s.prevalence_fail));
    m.insert("folds".into(), json!(s.params.folds));
    m.insert("repeats".into(), json!(s.params.repeats));
    m.insert("seed".into(), json!(s.params.seed));
    m.insert("bootstrap".into(), json!(s.params.bootstrap));
    m.insert(
        "arms".into(),
        Value::Object(
            s.arms
                .iter()
                .map(|a| (a.name.clone(), arm_json(a)))
                .collect(),
        ),
    );
    m.insert("gate".into(), gate_json(&s.gate));
    m.insert("notes".into(), json!(s.notes));
    m.insert("counts_by_source".into(), json!(s.counts_by_source));
    if let Some(d) = &s.dropped {
        m.insert(
            "dropped".into(),
            json!({"no_label": d.no_label, "no_readout": d.no_readout, "truncated": d.truncated, "no_answer": d.no_answer}),
        );
    }
    if !s.readout_meta.is_null() {
        m.insert("readout_meta".into(), s.readout_meta.clone());
    }
    if let Some(r) = &s.regime {
        m.insert("regime".into(), json!(r));
    }
    Value::Object(m)
}

/// The human-readable report.
pub fn markdown_report(res: &Study) -> String {
    let g = &res.gate;
    let mut l: Vec<String> = vec![
        "# Routing gate B0: AUROC of \"4B fails\"".to_string(),
        String::new(),
        format!(
            "Prompts: {} (failure rate {:.3}); per source {:?}.",
            res.n, res.prevalence_fail, res.counts_by_source
        ),
        format!(
            "Cross-validation: {}-fold, {} repeats, seed {}; bootstrap {}.",
            res.params.folds, res.params.repeats, res.params.seed, res.params.bootstrap
        ),
        String::new(),
        format!(
            "**Verdict: {}**. Gate arm `{}` AUROC {:.3} (95% CI {:.3} to {:.3}) against the baseline `{}` {:.3}; needs >= {:.2} and >= baseline + {:.2}. Point rule: {}; strict (CI) rule: {}.",
            g.verdict,
            g.arm,
            g.auroc,
            g.ci95[0],
            g.ci95[1],
            g.baseline_arm,
            g.baseline_auroc,
            g.threshold,
            g.margin,
            if g.point_pass { "met" } else { "NOT met" },
            if g.strict_pass { "met" } else { "NOT met" }
        ),
        format!("Releases: {}. If HOLD: {}.", g.releases, g.if_hold),
        String::new(),
        "| arm | AUROC | 95% CI | vs baseline (paired 95%) | within-source mean | ECE | Brier |"
            .to_string(),
        "|---|---|---|---|---|---|---|".to_string(),
    ];
    let mut order: Vec<&Arm> = res.arms.iter().collect();
    order.sort_by(|a, b| b.auroc.total_cmp(&a.auroc));
    for a in order {
        let diff = a.diff_vs_baseline.map_or_else(
            || "base".to_string(),
            |d| format!("{:+.3} ({:+.3} to {:+.3})", d.mean, d.lo, d.hi),
        );
        let ws = if a.within_source.is_empty() {
            "n/a".to_string()
        } else {
            format!(
                "{:.3}",
                a.within_source.values().sum::<f64>() / a.within_source.len() as f64
            )
        };
        l.push(format!(
            "| {} | {:.3} | {:.3} to {:.3} | {diff} | {ws} | {:.3} | {:.3} |",
            a.name, a.auroc, a.ci95[0], a.ci95[1], a.calibration.ece, a.calibration.brier
        ));
    }
    let per_source: Vec<String> = res
        .arm(GATE_ARM)
        .map(|a| {
            a.within_source
                .iter()
                .map(|(s, x)| format!("{s} {x:.3}"))
                .collect()
        })
        .unwrap_or_default();
    l.push(String::new());
    l.push(format!(
        "Per-source AUROC of the gate arm: {}",
        per_source.join(", ")
    ));
    l.push(String::new());
    if let Some(d) = &res.dropped {
        l.push(format!("Rows dropped before the study: {d:?}."));
        l.push(String::new());
    }
    l.push("Caveats:".to_string());
    l.extend(g.caveats.iter().map(|c| format!("- {c}")));
    if !res.notes.is_empty() {
        l.push(String::new());
        l.push("Notes:".to_string());
        l.extend(res.notes.iter().map(|c| format!("- {c}")));
    }
    l.push(String::new());
    l.push(
        "Orientation: scores mean \"more likely to fail\". Opposite-orientation AUROCs are in report.json (`flipped_auroc_diagnostic`) for diagnosis only and are never used for the gate."
            .to_string(),
    );
    l.join("\n") + "\n"
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// `report.json`, `report.md` and `calibration.csv` in `out_dir`.
pub fn write_outputs(res: &Study, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    std::fs::write(
        out_dir.join("report.json"),
        serde_json::to_string_pretty(&report_json(res))?,
    )?;
    std::fs::write(out_dir.join("report.md"), markdown_report(res))?;
    let mut csv = String::from("arm,bin,n,mean_pred,frac_fail,calibrated_by\n");
    for a in &res.arms {
        for (i, b) in a.calibration.bins.iter().enumerate() {
            writeln!(
                csv,
                "{},{i},{},{:.6},{:.6},{}",
                csv_field(&a.name),
                b.n,
                b.mean_pred,
                b.frac_pos,
                csv_field(&a.calibration.calibrated_by)
            )?;
        }
    }
    std::fs::write(out_dir.join("calibration.csv"), csv)?;
    Ok(())
}

/// `oof_scores.csv`: one row per prompt with every arm's out-of-fold score.
pub fn write_oof(res: &Study, out_dir: &Path) -> Result<()> {
    let mut names: Vec<&Arm> = res.arms.iter().collect();
    names.sort_by(|a, b| a.name.cmp(&b.name));
    let mut csv = String::from("id,source,fails");
    for a in &names {
        csv.push(',');
        csv.push_str(&csv_field(&a.name));
    }
    csv.push('\n');
    for i in 0..res.n {
        write!(
            csv,
            "{},{},{}",
            csv_field(&res.ids[i]),
            csv_field(&res.sources[i]),
            u8::from(res.y[i])
        )?;
        for a in &names {
            write!(csv, ",{:.6}", a.score[i])?;
        }
        csv.push('\n');
    }
    std::fs::write(out_dir.join("oof_scores.csv"), csv)?;
    Ok(())
}

/// Everything: [`write_outputs`] and [`write_oof`].
pub fn write_outputs_with_oof(res: &Study, out_dir: &Path) -> Result<()> {
    write_outputs(res, out_dir)?;
    write_oof(res, out_dir)
}
