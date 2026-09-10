// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the concurrency sweep: cell layout, fixture selection, the
//! vacuity floor, and the metrics map future gating reads.
//!
//! The run-verdict half lives in `concurrency_verdict_tests.rs` — split for
//! the 500-LoC cap when the ladder widened to C=128, which roughly doubled
//! the rung tables both halves assert on.

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
    // Default to the MTP arm: `accepted` such that accept_len is ~2.3, this
    // model's measured accept depth. Existing tests assert on comparable
    // cells, and a cell that drew the SERIAL arm is deliberately not
    // comparable — so a serial default would silently gut them.
    evidence_with_arm(completion_tokens, Some(completion_tokens * 13 / 23))
}

/// Evidence with an explicit speculation arm. `None` = the server reported no
/// accept field at all, which must NOT be read as serial.
fn evidence_with_arm(
    completion_tokens: usize,
    accepted_prediction_tokens: Option<usize>,
) -> RequestEvidence {
    RequestEvidence {
        completion_tokens,
        prompt_tokens: 512,
        cached_prompt_tokens: 512,
        finish_reason: Some("length".into()),
        server_ttft_ms: None,
        server_tps: None,
        accepted_prediction_tokens,
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
        cache_uncontrolled: false,
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

/// The property that makes a CONCURRENT warm-up round safe: within one round
/// every tag is distinct, so no two in-flight warm-up requests target the same
/// prefix-cache key. If a future plan ever repeated a tag inside a round, the
/// round would race two inserts on one key and the "warmed" claim would become
/// timing-dependent — so this is asserted rather than assumed.
#[test]
fn a_warmup_round_never_repeats_a_prompt() {
    for conc in [1usize, 2, 4, 16, 128] {
        let plan = prompt_plan(conc, 1);
        for round in &plan.warmup_rounds {
            let mut seen = std::collections::BTreeSet::new();
            for tag in round {
                assert!(
                    seen.insert(tag.clone()),
                    "conc {conc}: warm-up round repeats {tag}, so its requests \
                     would race on one cache key"
                );
            }
            assert_eq!(seen.len(), conc, "conc {conc}: round is not the full set");
        }
    }
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

#[test]
fn vacuity_flags_any_request_below_80_pct_of_osl() {
    let osl = 100;
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

#[test]
fn a_requested_warm_path_requires_observed_cached_tokens() {
    let osl = 128;
    let mut missed = row(4, 100.0, Some(50.0), vec![evidence(128); 4], osl);
    missed.requests[2].cached_prompt_tokens = 0;
    missed.cache_uncontrolled = cache_is_uncontrolled(&missed.requests, 1);

    assert!(missed.cache_uncontrolled);
    assert!(!missed.comparable());
    assert_eq!(missed.min_cached_prompt(), Some(0));
    assert_eq!(missed.min_cached_prompt_pct(), Some(0.0));
    assert!(
        evidence_line(512, 4, &missed.requests).contains("cached [512/512,512/512,0/512,512/512]")
    );

    assert!(!cache_is_uncontrolled(&missed.requests, 0));
    assert!(!cache_is_uncontrolled(&[evidence(128), evidence(128)], 1));

    let mut boundary = evidence(128);
    boundary.prompt_tokens = 100;
    boundary.cached_prompt_tokens = 80;
    assert!(!cache_is_uncontrolled(&[boundary.clone()], 1));
    boundary.cached_prompt_tokens = 79;
    assert!(cache_is_uncontrolled(&[boundary], 1));

    let mut missing_usage = evidence(128);
    missing_usage.prompt_tokens = 0;
    missing_usage.cached_prompt_tokens = 0;
    assert!(cache_is_uncontrolled(&[missing_usage], 1));

    let mut b = configured(vec![4], vec![512]);
    b.rows.push(missed);
    let table = b.table();
    assert_eq!(table.columns[10].title, "min cache%");
    assert_eq!(table.rows[0][8].style, CellStyle::Bad);
    assert_eq!(table.rows[0][8].text, "100.0*");
    assert_eq!(table.rows[0][10].text, "0");
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
    assert_eq!(m.get("min_completion_tokens"), Some(&1.0));
    assert_eq!(m.get("min_cached_prompt_tokens"), Some(&512.0));
    assert_eq!(m.get("min_cached_prompt_pct"), Some(&100.0));
    assert_eq!(m.get("vacuous_cells"), Some(&1.0));
    assert_eq!(m.get("cache_uncontrolled_cells"), Some(&0.0));
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

// ── THE ARM PIN (#835) ───────────────────────────────────────────────────
//
// `c2_aggregate_tok_s` is trimodal (~30.6 / ~27.5 / ~23.5 tok/s, nothing
// between) because the C=2 cell's ~640 measured tokens contain only one to
// three MTP-gate arbitrations, so its outcome is all-MTP, mixed, or
// all-serial. The committed floor sits in the empty gap, which makes it a
// mode detector rather than a regression detector.
//
// ★ CORRECTED after a real campaign. The arm was first wired as a COMPARABILITY
// class — a serial cell excluded from scoring, and any such cell failing the run
// INCONCLUSIVE. That was wrong, and only a live gate run showed it: at wide batch
// the MTP gate drops speculation ON PURPOSE, so C=8 upward legitimately run
// serial, and the floors for those rungs were calibrated on runs that did
// exactly that. The pin dropped five of eight cells, made `peak_aggregate_tok_s`
// read 49.5 (from C=4) instead of ~115 (from C=64), and failed a gate with nine
// consecutive passing records.
//
// The arm is now PUBLISHED and never gated on: the benchmark cannot know which
// arm should have run, and asserting otherwise is a claim it cannot support.
// What #835 needs is that a human reading a low C=2 can see whether the serial
// arm explains it — and a metric on the record does that.

#[test]
fn a_serial_arm_cell_is_reported_but_still_scored() {
    // accept_len == 1.00 exactly: every emitted token cost one step.
    let serial = row(
        2,
        23.5,
        Some(2000.0),
        vec![evidence_with_arm(320, Some(0)); 2],
        320,
    );
    assert_eq!(serial.accept_len(), Some(1.0));
    assert!(serial.arm_is_not_mtp(), "accept_len 1.00 is the serial arm");
    // ★ THE CORRECTION. It stays comparable. At wide batch the gate drops
    // speculation deliberately and those floors were calibrated that way, so
    // excluding a serial cell throws away a measurement the floor expects.
    assert!(
        serial.comparable(),
        "a serial cell is still comparable to a floor calibrated on serial"
    );
    assert!(!serial.vacuous);
    assert!(!serial.cache_uncontrolled);
}

#[test]
fn an_mtp_arm_cell_is_comparable() {
    let mtp = row(2, 30.6, Some(2000.0), vec![evidence(320); 2], 320);
    let a = mtp.accept_len().expect("accept_len derivable");
    assert!(
        a > 1.5,
        "the MTP arm sits well above the 1.5 threshold, got {a}"
    );
    assert!(!mtp.arm_is_not_mtp());
    assert!(mtp.comparable());
}

#[test]
fn a_mixed_cell_reports_the_minimum_arm_and_is_still_scored() {
    // One request served serial, one speculative. The MINIMUM governs the
    // reported figure, so a cell that touched the serial arm at all says so —
    // that is the signal a human needs to explain a low reading.
    let mixed = row(
        2,
        27.5,
        Some(2000.0),
        vec![evidence_with_arm(320, Some(0)), evidence(320)],
        320,
    );
    assert_eq!(mixed.accept_len(), Some(1.0), "the minimum, not the mean");
    assert!(
        mixed.comparable(),
        "reporting the arm must not remove the cell from scoring"
    );
}

#[test]
fn a_missing_accept_field_is_not_read_as_serial() {
    // ★ THE TRAP. `None` means the server did not report the field. Treating
    // it as 1.00 would convert a missing instrument into a verdict about the
    // engine — and every pre-existing recorded sweep has no accept field.
    let unknown = row(
        2,
        27.5,
        Some(2000.0),
        vec![evidence_with_arm(320, None); 2],
        320,
    );
    assert_eq!(unknown.accept_len(), None);
    assert!(
        !unknown.arm_is_not_mtp(),
        "an unreported arm is unknown, not serial"
    );
    assert!(
        unknown.comparable(),
        "a sweep from before this instrument existed must stay comparable"
    );
}

#[test]
fn a_corrupt_accept_count_does_not_divide_by_zero() {
    // accepted >= completion is impossible on the wire but must not panic or
    // produce a negative/infinite accept depth if it ever appears.
    let corrupt = row(
        2,
        27.5,
        Some(2000.0),
        vec![evidence_with_arm(320, Some(320)); 2],
        320,
    );
    assert_eq!(corrupt.accept_len(), None);
    assert!(!corrupt.arm_is_not_mtp());
}
