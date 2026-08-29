// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the concurrency sweep: cell layout, fixture selection, the
//! vacuity floor, and the metrics map future gating reads.

use super::*;

fn configured(concs: Vec<i64>, isls: Vec<i64>) -> ConcurrencySweep {
    let mut b = ConcurrencySweep::default();
    let mut v = ParamValues::defaults(&b.parameters());
    v.set("concurrencies", ParamValue::IntList(concs));
    v.set("isls", ParamValue::IntList(isls));
    b.configure(&v).unwrap();
    b
}

fn evidence(completion_tokens: usize) -> RequestEvidence {
    RequestEvidence {
        completion_tokens,
        finish_reason: Some("length".into()),
        server_ttft_ms: None,
        server_tps: None,
    }
}

fn row(
    conc: usize,
    throughput: f64,
    ttft_p50: Option<f64>,
    requests: Vec<RequestEvidence>,
    osl: usize,
) -> CellRow {
    let vacuous = cell_is_vacuous(&requests, osl);
    CellRow {
        isl: 512,
        conc,
        ttft: Percentiles {
            p50: ttft_p50,
            p90: ttft_p50,
            p99: ttft_p50,
        },
        tpot: Percentiles::default(),
        e2e_p50: None,
        throughput,
        errors: 0,
        requests,
        vacuous,
    }
}

#[test]
fn cells_are_isl_major() {
    let b = configured(vec![1, 2], vec![128, 512]);
    assert_eq!(b.cells, vec![(128, 1), (128, 2), (512, 1), (512, 2)]);
}

#[test]
fn defaults_are_the_campaign_sweep() {
    let b = ConcurrencySweep::default();
    let v = ParamValues::defaults(&b.parameters());
    // The PARAM DEFAULTS are what actually runs — `configure()` rebuilds
    // from these, so a descriptor blurb saying "1 → 32" proves nothing on
    // its own. This assertion is the only thing that pins the sweep.
    assert_eq!(v.int_list("concurrencies").unwrap(), &[1, 2, 4, 8, 16, 32]);
    assert_eq!(v.int_list("isls").unwrap(), &[128, 512, 1024, 2048]);
    assert_eq!(v.usize("osl").unwrap(), 128);
    assert_eq!(v.usize("warmup").unwrap(), 1);
    assert_eq!(v.text("prompt_mode").unwrap(), "natural");
    assert_eq!(v.usize("request_timeout_s").unwrap(), 600);
}

#[test]
fn an_out_of_range_parameter_is_rejected_before_the_run() {
    let mut b = ConcurrencySweep::default();
    let mut v = ParamValues::defaults(&b.parameters());
    v.set("osl", ParamValue::Int(0));
    let err = b.configure(&v).unwrap_err().to_string();
    assert!(err.contains("Output tokens"), "{err}");
}

#[test]
fn reconfiguring_clears_prior_rows() {
    let mut b = configured(vec![1], vec![128]);
    b.rows.push(CellRow::default());
    let mut v = ParamValues::defaults(&b.parameters());
    v.set("isls", ParamValue::IntList(vec![256]));
    b.configure(&v).unwrap();
    assert!(b.rows.is_empty() && b.cursor == 0);
}

#[test]
fn warmup_and_measurement_share_the_complete_prompt_set() {
    let b = configured(vec![4], vec![512]);
    let plan = prompt_plan(4, 2);

    assert!(prompt_plan(4, 0).warmup_rounds.is_empty());
    assert_eq!(plan.measured, ["c0", "c1", "c2", "c3"]);
    assert_eq!(
        plan.warmup_rounds,
        [plan.measured.clone(), plan.measured.clone()]
    );
    let warmup_prompts: Vec<_> = plan.warmup_rounds[0]
        .iter()
        .map(|tag| b.cell_prompt(512, tag))
        .collect();
    let measured_prompts: Vec<_> = plan
        .measured
        .iter()
        .map(|tag| b.cell_prompt(512, tag))
        .collect();

    assert_eq!(warmup_prompts, measured_prompts);
    assert_ne!(b.cell_prompt(512, "warm"), measured_prompts[0]);
}

// ---- fixture selection ------------------------------------------------------

/// The default is the natural code-generation fixture — the 2026-08-15
/// re-scope. The counting prompt produced 49-token bursts at C=1 and 0–1
/// token cells at C≥4 on a serve where a MinHeap-class prompt completed the
/// full budget at every C; defaulting to count would ship the broken
/// instrument.
#[test]
fn default_prompt_mode_is_the_code_generation_fixture() {
    let b = configured(vec![1], vec![512]);
    assert_eq!(b.mode, PromptMode::Natural);
    let p = b.cell_prompt(512, "c0");
    assert!(
        p.contains("MinHeap"),
        "natural mode must pose the code task"
    );
    assert!(p.ends_with(CODE_TASK));
    use sha2::{Digest, Sha256};
    assert_eq!(
        format!("{:x}", Sha256::digest(CODE_TASK.as_bytes())),
        "7f51f5f271897f32801928c01b59f49472ad3e1880366fccb3cef4cb79db56cd"
    );
    // Padding still tracks the ISL: a bigger request means a longer prompt.
    assert!(b.cell_prompt(2048, "c0").len() > p.len());
}

#[test]
fn count_mode_still_appends_the_counting_instruction() {
    let mut b = configured(vec![1], vec![512]);
    let mut v = ParamValues::defaults(&b.parameters());
    v.set("prompt_mode", ParamValue::Text("count".into()));
    b.configure(&v).unwrap();
    let p = b.cell_prompt(512, "c0");
    assert!(p.ends_with("until told to stop."));
    assert!(!p.contains("MinHeap"));
}

/// The old help text claimed "count forces the full output budget so TPOT is
/// real". That was measured FALSE (49-token bursts under count mode), and the
/// server has no ignore_eos, so no mode can force the budget. The help must
/// not resurrect the claim.
#[test]
fn prompt_mode_help_does_not_claim_to_force_the_budget() {
    let b = ConcurrencySweep::default();
    let spec = b
        .parameters()
        .into_iter()
        .find(|s| s.key == "prompt_mode")
        .expect("prompt_mode param");
    assert!(
        !spec.help.contains("forces the full output budget"),
        "help text resurrects the disproven forcing claim: {}",
        spec.help
    );
    assert!(
        spec.help.contains("vacuous"),
        "help must point at the vacuity floor"
    );
}

// ---- vacuity floor ----------------------------------------------------------

#[test]
fn vacuity_flags_any_request_below_80_pct_of_osl() {
    let osl = 100;
    // Exactly at the floor is NOT vacuous (a natural stop a few tokens early
    // is fine); one token below it is.
    assert!(!cell_is_vacuous(&[evidence(80), evidence(100)], osl));
    assert!(cell_is_vacuous(&[evidence(79), evidence(100)], osl));
    // ONE short request poisons the whole cell — its wall time is in the
    // denominator of the aggregate.
    assert!(cell_is_vacuous(
        &[evidence(100), evidence(100), evidence(0)],
        osl
    ));
    // No successful requests: nothing to call vacuous (the error count
    // already invalidates the cell).
    assert!(!cell_is_vacuous(&[], osl));
}

#[test]
fn a_vacuous_cell_is_not_comparable_and_an_errored_cell_is_not_either() {
    let osl = 128;
    let good = row(4, 100.0, Some(50.0), vec![evidence(128); 4], osl);
    let short = row(4, 400.0, Some(10.0), vec![evidence(1); 4], osl);
    let mut errored = row(4, 90.0, Some(50.0), vec![evidence(128); 3], osl);
    errored.errors = 1;
    assert!(good.comparable());
    assert!(short.vacuous && !short.comparable());
    assert!(!errored.comparable());
}

// ---- metrics map ------------------------------------------------------------

/// The sweep previously emitted NO metrics at all — nothing for a future
/// gate to compare. This pins the map's presence and its exclusion rule:
/// vacuous cells must never mint a throughput number, however large their
/// (bogus) tok/s is, while min_completion_tokens spans every request because
/// it is the evidence behind the exclusion.
#[test]
fn metrics_map_reports_the_comparable_curve_and_the_evidence_floor() {
    let osl = 128;
    let mut b = configured(vec![1, 4], vec![512]);
    b.osl = osl;
    b.rows
        .push(row(1, 30.0, Some(120.0), vec![evidence(128)], osl));
    b.rows
        .push(row(4, 100.0, Some(150.0), vec![evidence(128); 4], osl));
    // A later, lower comparable row must not replace the best C=4 cell.
    b.rows
        .push(row(4, 80.0, Some(300.0), vec![evidence(128); 4], osl));
    // A vacuous C=4 cell with a huge bogus rate: must not win the rung.
    b.rows
        .push(row(4, 900.0, Some(5.0), vec![evidence(1); 4], osl));
    let m = b.metrics();
    assert_eq!(m.get("c1_aggregate_tok_s"), Some(&30.0));
    assert_eq!(m.get("c4_aggregate_tok_s"), Some(&100.0));
    assert_eq!(m.get("c1_ttft_p50_ms"), Some(&120.0));
    assert_eq!(m.get("c4_ttft_p50_ms"), Some(&150.0));
    assert_eq!(m.get("peak_aggregate_tok_s"), Some(&100.0));
    // The floor sees THROUGH the exclusion: the 1-token requests are the
    // record of what went wrong.
    assert_eq!(m.get("min_completion_tokens"), Some(&1.0));
    assert_eq!(m.get("vacuous_cells"), Some(&1.0));
}

#[test]
fn metrics_map_with_no_comparable_cells_still_reports_evidence() {
    let osl = 128;
    let mut b = configured(vec![1], vec![512]);
    b.osl = osl;
    b.rows
        .push(row(1, 500.0, Some(10.0), vec![evidence(0)], osl));
    let m = b.metrics();
    assert!(!m.contains_key("peak_aggregate_tok_s"));
    assert!(!m.contains_key("c1_aggregate_tok_s"));
    assert_eq!(m.get("min_completion_tokens"), Some(&0.0));
    assert_eq!(m.get("vacuous_cells"), Some(&1.0));
}

// ---- self-verdict (C1 pattern, gate promotion 2026-08-15) -------------------

use super::verdict::{Floors, sweep_verdict};
use crate::result::VerdictKind;

fn floors(c1: f64, c4: f64, c8: f64, c16: f64, peak: f64) -> Floors {
    Floors {
        per_c: vec![(1, c1), (4, c4), (8, c8), (16, c16)],
        peak,
    }
}

fn ladder(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
    entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

fn committed_floors() -> Floors {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace layout");
    let (_, entry) = crate::gate::bench::load_all(root)
        .expect("committed BENCH.toml files must load")
        .into_iter()
        .find(|(target, entry)| {
            target.hardware == "gb10"
                && target.model == "qwen3.8-27b"
                && entry.checkpoint == "unsloth/Qwen3.8-27B-NVFP4"
                && entry.gate == "concurrency-sweep"
        })
        .expect("the measured Qwen3.8 concurrency gate must be committed");
    let metrics = entry
        .metrics
        .expect("the measured concurrency gate must have bounds");
    let min = |metric: &str| {
        metrics[metric]
            .min
            .unwrap_or_else(|| panic!("{metric} must have a minimum"))
    };
    Floors {
        per_c: vec![
            (1, min("c1_aggregate_tok_s")),
            (4, min("c4_aggregate_tok_s")),
            (8, min("c8_aggregate_tok_s")),
            (16, min("c16_aggregate_tok_s")),
        ],
        peak: min("peak_aggregate_tok_s"),
    }
}

/// The calibrated ladder against its committed floors — the PASS the gate
/// machinery requires now that the sweep is REQUIRED. The reason names every
/// gated rung so the record reads as evidence, not a bare verdict.
#[test]
fn a_clean_sweep_that_clears_every_floor_passes() {
    let m = ladder(&[
        ("c1_aggregate_tok_s", 25.5),
        ("c4_aggregate_tok_s", 47.5),
        ("c8_aggregate_tok_s", 67.6),
        ("c16_aggregate_tok_s", 98.9),
        ("peak_aggregate_tok_s", 98.9),
    ]);
    let floors = committed_floors();
    assert_eq!(
        floors.per_c,
        vec![(1, 17.0), (4, 35.0), (8, 52.0), (16, 73.5)]
    );
    assert_eq!(floors.peak, 73.5);
    let v = sweep_verdict(&m, 4, 0, 0, 80.0, &floors);
    assert_eq!(v.kind, VerdictKind::Pass, "{}", v.reason);
    for rung in ["C1", "C4", "C8", "C16", "peak"] {
        assert!(v.reason.contains(rung), "{}", v.reason);
    }
}

/// FAIL names the violating cell — and the comparison is the raw value
/// against the floor, deliberately stricter than gate scoring's
/// value + noise >= min.
#[test]
fn a_sweep_below_one_floor_fails_naming_the_cell() {
    let committed = committed_floors();
    let c8_floor = committed.per_c[2].1;
    let m = ladder(&[
        ("c1_aggregate_tok_s", 25.5),
        ("c4_aggregate_tok_s", 47.5),
        ("c8_aggregate_tok_s", 51.2),
        ("c16_aggregate_tok_s", 98.9),
        ("peak_aggregate_tok_s", 98.9),
    ]);
    let v = sweep_verdict(&m, 4, 0, 0, 80.0, &committed);
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("C=8"), "{}", v.reason);
    assert!(
        v.reason.contains("51.2") && v.reason.contains("52.0"),
        "{}",
        v.reason
    );
    // Exactly on the floor passes — inclusive, like the BENCH.toml bound.
    let m = ladder(&[("c8_aggregate_tok_s", c8_floor)]);
    let v = sweep_verdict(&m, 1, 0, 0, 80.0, &floors(0.0, 0.0, c8_floor, 0.0, 0.0));
    assert_eq!(v.kind, VerdictKind::Pass, "{}", v.reason);
}

/// Floors all zero (the schema default) keep the pre-gate info verdicts, both
/// arms verbatim: a standalone sweep has no committed ladder to be judged
/// against.
#[test]
fn all_floors_zero_keeps_the_info_verdicts() {
    let m = ladder(&[("c1_aggregate_tok_s", 25.5)]);
    let clean = sweep_verdict(&m, 4, 0, 0, 80.0, &Floors::default());
    assert_eq!(clean.kind, VerdictKind::Info, "{}", clean.reason);
    assert!(
        clean.reason.contains("no request errors"),
        "{}",
        clean.reason
    );
    let vac = sweep_verdict(&m, 4, 0, 2, 80.0, &Floors::default());
    assert_eq!(vac.kind, VerdictKind::Info, "{}", vac.reason);
    assert!(vac.reason.contains("not comparable"), "{}", vac.reason);
}

/// ★ Vacuous cells can NEVER pass a gating sweep, whatever the numbers say:
/// the aggregate divides undelivered tokens' wall time into real tokens. This
/// is the rule the floors cannot override.
#[test]
fn vacuous_cells_fail_a_gating_sweep_regardless_of_the_floors() {
    let m = ladder(&[
        ("c1_aggregate_tok_s", 999.0),
        ("c4_aggregate_tok_s", 999.0),
        ("c8_aggregate_tok_s", 999.0),
        ("c16_aggregate_tok_s", 999.0),
        ("peak_aggregate_tok_s", 999.0),
    ]);
    let v = sweep_verdict(&m, 4, 0, 1, 80.0, &floors(24.0, 43.0, 63.0, 94.0, 94.0));
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("INCONCLUSIVE"), "{}", v.reason);
    assert!(v.reason.contains("vacuity floor"), "{}", v.reason);
}

/// A gated rung the sweep never measured comparably must not pass by
/// omission: the floor demands the measurement itself.
#[test]
fn a_gated_rung_with_no_comparable_cell_fails_as_inconclusive() {
    // C=16 gated but absent from the metrics (its only cell was excluded).
    let m = ladder(&[("c1_aggregate_tok_s", 25.5)]);
    let v = sweep_verdict(&m, 4, 0, 0, 80.0, &floors(0.0, 0.0, 0.0, 94.0, 0.0));
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("C=16"), "{}", v.reason);
    assert!(v.reason.contains("INCONCLUSIVE"), "{}", v.reason);
    // Same for the peak floor.
    let v = sweep_verdict(&m, 4, 0, 0, 80.0, &floors(0.0, 0.0, 0.0, 0.0, 94.0));
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("peak"), "{}", v.reason);
}

/// Request errors fail the sweep gating or not — an errored cell's numbers
/// are not comparable, and the floors cannot buy them back.
#[test]
fn request_errors_fail_the_sweep_in_both_modes() {
    let m = ladder(&[("c1_aggregate_tok_s", 999.0)]);
    for f in [Floors::default(), floors(24.0, 43.0, 63.0, 94.0, 94.0)] {
        let v = sweep_verdict(&m, 4, 2, 0, 80.0, &f);
        assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
        assert!(v.reason.contains("2 request(s) failed"), "{}", v.reason);
    }
}

/// The descriptor couples each floor param to the metric its BENCH.toml bound
/// is written on, every param exists in the schema with the documented OFF
/// default, and `configure` carries the values into the verdict floors.
#[test]
fn the_floor_params_are_wired_to_the_gate() {
    assert_eq!(
        DESCRIPTOR.threshold_params,
        [
            ("min_c1", "c1_aggregate_tok_s"),
            ("min_c4", "c4_aggregate_tok_s"),
            ("min_c8", "c8_aggregate_tok_s"),
            ("min_c16", "c16_aggregate_tok_s"),
            ("min_peak", "peak_aggregate_tok_s"),
        ]
    );
    let mut b = ConcurrencySweep::default();
    let specs = b.parameters();
    for (param, _) in DESCRIPTOR.threshold_params {
        assert!(
            specs.iter().any(|s| s.key == *param),
            "{param} declared but missing from the schema"
        );
    }
    let mut v = ParamValues::defaults(&specs);
    b.configure(&v).unwrap();
    assert!(!b.floors.gating(), "defaults must not gate");
    v.set("min_c8", ParamValue::Float(63.0));
    v.set("min_peak", ParamValue::Float(94.0));
    b.configure(&v).unwrap();
    assert!(b.floors.gating());
    assert_eq!(
        b.floors.per_c,
        vec![(1, 0.0), (4, 0.0), (8, 63.0), (16, 0.0)]
    );
    assert_eq!(b.floors.peak, 94.0);
}
