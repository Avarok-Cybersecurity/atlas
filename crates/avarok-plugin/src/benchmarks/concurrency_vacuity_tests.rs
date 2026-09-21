// SPDX-License-Identifier: AGPL-3.0-only

//! The vacuity rule against the two things it is judged by: the documented
//! failure it was invented for (2026-08-15: 49-token bursts at C=1, 0–1-token
//! cells at C≥4, aggregate decreasing with C) and the published instrument's
//! own receipts (`bench/ladder38/l38_r11_*.json`, ISL 128 / OSL 1024), which
//! the old per-request rule voided reproducibly.
//!
//! Every fixture that is meant to be the case the old rule refused ASSERTS
//! that it is — a fixture that no longer contains a request under 80% would
//! pass the new rule for the wrong reason.

use super::concurrency_tests::{cell_is_vacuous, evidence, row};
use super::vacuity::{Delivery, VACUITY_FLOOR};
use super::*;

/// The ladder38 output budget and the per-request bar the old rule applied
/// to it (0.8 × 1024).
const OSL: usize = 1024;
const OLD_BAR: usize = 820;

/// The old rule, kept here as the reference the receipts are contrasted
/// against: ANY request under the bar voided the cell.
fn old_rule_voids(completions: &[usize], osl: usize) -> bool {
    completions
        .iter()
        .any(|&t| (t as f64) < VACUITY_FLOOR * osl as f64)
}

fn cell(completions: &[usize]) -> Vec<RequestEvidence> {
    completions.iter().map(|&t| evidence(t)).collect()
}

/// `completion_tokens_per_req` of every rep of a committed ladder38 receipt.
fn receipt_reps(json: &str) -> Vec<Vec<usize>> {
    let v: serde_json::Value = serde_json::from_str(json).expect("receipt parses");
    assert_eq!(v["osl"].as_u64(), Some(OSL as u64), "receipt osl");
    let rungs = v["rungs"].as_array().expect("rungs");
    assert_eq!(rungs.len(), 1, "one rung per receipt");
    rungs[0]["reps"]
        .as_array()
        .expect("reps")
        .iter()
        .map(|rep| {
            rep["completion_tokens_per_req"]
                .as_array()
                .expect("completion_tokens_per_req")
                .iter()
                .map(|t| t.as_u64().expect("token count") as usize)
                .collect()
        })
        .collect()
}

const R11_C8: &str = include_str!("../../../../bench/ladder38/l38_r11_c8.json");
const R11_C64: &str = include_str!("../../../../bench/ladder38/l38_r11_c64.json");
const R11_C128: &str = include_str!("../../../../bench/ladder38/l38_r11_c128.json");

// ── the failure the rule was invented for ──────────────────────────────────

/// C=1, 49 tokens against a 512-token budget: 9.6% delivered, and the one
/// request is the median. Vacuous on both clauses.
#[test]
fn the_documented_c1_burst_is_vacuous() {
    let d = Delivery::of(&cell(&[49]), 512);
    assert_eq!((d.delivered, d.budget, d.cleared), (49, 512, 0));
    assert!(d.is_vacuous(), "{}", d.describe(512));
    assert!(cell_is_vacuous(&cell(&[49]), 512));
}

/// C≥4 cells of 0–1 output tokens: E2E==TTFT, TPOT unmeasurable (the client
/// reports no TPOT under two tokens, so the percentile is `None`), and the
/// aggregate falling with C. Each cell is vacuous, none is comparable, and
/// the ladder mints no throughput at any rung.
#[test]
fn the_documented_no_decode_ladder_is_vacuous_at_every_rung() {
    let osl = 512;
    let burst = row(1, 30.0, Some(120.0), cell(&[49]), osl);
    let mut c4 = row(4, 8.0, Some(150.0), cell(&[0, 1, 1, 0]), osl);
    c4.e2e_p50 = c4.ttft.p50;
    let mut c16 = row(16, 2.0, Some(400.0), cell(&[1; 16]), osl);
    c16.e2e_p50 = c16.ttft.p50;
    for r in [&burst, &c4, &c16] {
        assert!(r.vacuous, "C={} must be vacuous", r.conc);
        assert!(!r.comparable(), "C={} must not be comparable", r.conc);
        assert!(r.tpot.p50.is_none(), "no decode interval to time");
    }
    assert_eq!(c4.e2e_p50, c4.ttft.p50, "E2E == TTFT: nothing was decoded");

    let sweep = ConcurrencySweep {
        osl,
        rows: vec![burst, c4, c16],
        ..Default::default()
    };
    let m = sweep.metrics();
    for c in [1, 4, 16] {
        assert!(
            !m.contains_key(&format!("c{c}_aggregate_tok_s")),
            "a vacuous C={c} cell minted a throughput number"
        );
    }
    assert!(!m.contains_key("peak_aggregate_tok_s"));
    assert_eq!(m.get("vacuous_cells"), Some(&3.0));
    assert_eq!(m.get("min_completion_tokens"), Some(&0.0));
}

// ── the published instrument ───────────────────────────────────────────────

/// The C=64 and C=128 cells of round 11, straight from the receipts. Rep 2
/// of each holds a natural stop (753 and 691 tokens) that the old rule
/// refused; the rule must call none of the six cells vacuous.
#[test]
fn the_published_c64_and_c128_cells_are_not_vacuous() {
    for (json, conc, natural_stop) in [(R11_C64, 64, 753), (R11_C128, 128, 691)] {
        let reps = receipt_reps(json);
        assert_eq!(reps.len(), 3, "C={conc}: three reps");
        for (i, completions) in reps.iter().enumerate() {
            assert_eq!(
                completions.len(),
                conc,
                "C={conc} rep {}: one entry per request",
                i + 1
            );
            let d = Delivery::of(&cell(completions), OSL);
            assert!(
                !d.is_vacuous(),
                "C={conc} rep {}: the published cell must not be vacuous: {}",
                i + 1,
                d.describe(OSL)
            );
            assert!(
                d.delivered_pct() > 99.0,
                "C={conc} rep {}: {}",
                i + 1,
                d.describe(OSL)
            );
        }
        // The relief is real only if the fixture contains what the old rule
        // refused: rep 2's minimum is the natural stop and it sits under
        // the old per-request bar.
        let rep2 = &reps[1];
        assert_eq!(
            rep2.iter().copied().min(),
            Some(natural_stop),
            "C={conc} rep 2 minimum"
        );
        assert!(natural_stop < OLD_BAR);
        assert!(
            old_rule_voids(rep2, OSL),
            "C={conc} rep 2: the old rule voided this cell"
        );
        assert_eq!(
            rep2.iter().filter(|&&t| t < OLD_BAR).count(),
            1,
            "C={conc} rep 2: exactly one request under the old bar"
        );
        assert!(!old_rule_voids(&reps[0], OSL) && !old_rule_voids(&reps[2], OSL));
    }
}

/// C=8 rep 3 of the same round: one request at 745 of eight, 96.6%
/// delivered. The smallest rung where the receipts show a natural stop.
#[test]
fn the_published_c8_cell_with_a_745_token_stop_is_not_vacuous() {
    let reps = receipt_reps(R11_C8);
    let rep3 = &reps[2];
    assert_eq!(rep3.iter().copied().min(), Some(745));
    assert!(old_rule_voids(rep3, OSL));
    let d = Delivery::of(&cell(rep3), OSL);
    assert_eq!((d.requests, d.cleared), (8, 7));
    assert!(!d.is_vacuous(), "{}", d.describe(OSL));
    for completions in &reps {
        assert!(!cell_is_vacuous(&cell(completions), OSL));
    }
}

// ── the rule's edges: each clause proven load-bearing ──────────────────────

/// Occupancy is judged on the cell's TOTAL: at the floor passes, one token
/// under it fails, however the shortfall is spread.
#[test]
fn occupancy_is_judged_at_the_floor_on_the_cell_total() {
    // 4 × 1024 = 4096 budget; 80% = 3276.8, so 3277 delivered clears it.
    let at_floor = [1024, 1024, 1024, 205];
    assert_eq!(at_floor.iter().sum::<usize>(), 3277);
    assert!(!cell_is_vacuous(&cell(&at_floor), OSL));
    let one_under = [1024, 1024, 1024, 204];
    assert!(cell_is_vacuous(&cell(&one_under), OSL));
    // Same total shortfall spread across two requests: same verdict.
    assert!(cell_is_vacuous(&cell(&[1024, 1024, 614, 614]), OSL));
    // EXACTLY at the floor (0.8 × 5120 = 4096 delivered): not vacuous. The
    // 1024 budget never lands on the boundary, so this is where a `<` that
    // drifts to `<=` shows.
    assert!(!cell_is_vacuous(&cell(&[1024, 1024, 1024, 1024, 0]), OSL));
    // And exactly at both bars at once: two requests at 80 of 100.
    assert!(!cell_is_vacuous(&cell(&[80, 80]), 100));
}

/// The majority clause is load-bearing on its own: a cell whose TOTAL clears
/// 80% is still vacuous when half or more of its requests fall under the
/// per-request bar, because the p50s would then be read from short requests.
#[test]
fn a_majority_under_the_bar_is_vacuous_even_when_the_total_clears_it() {
    // 3 × 800 + 1024 = 3424 of 4096 = 83.6% delivered, 1 of 4 cleared.
    let d = Delivery::of(&cell(&[800, 800, 800, 1024]), OSL);
    assert!(d.delivered_pct() > 80.0);
    assert_eq!(d.cleared, 1);
    assert!(d.is_vacuous(), "{}", d.describe(OSL));
    // Exactly half cleared is enough: the nearest-rank p50 of 4 is index 2,
    // the first of the two full requests.
    let half = Delivery::of(&cell(&[800, 800, 1024, 1024]), OSL);
    assert_eq!(half.cleared, 2);
    assert!(!half.is_vacuous(), "{}", half.describe(OSL));
    // Odd n: 1 short of 3 passes, 2 short of 3 does not.
    assert!(!cell_is_vacuous(&cell(&[800, 1024, 1024]), OSL));
    assert!(cell_is_vacuous(&cell(&[819, 819, 1024]), OSL));
}

/// At C=1 the cell is the request, so the rule is the old rule: the 691-token
/// natural stop that a C=128 cell absorbs is vacuous on its own.
#[test]
fn at_c1_the_rule_is_the_old_rule() {
    // 80 of 100 is exactly the bar and clears it.
    assert!(!cell_is_vacuous(&cell(&[80]), 100));
    assert!(cell_is_vacuous(&cell(&[79]), 100));
    assert!(cell_is_vacuous(&cell(&[691]), OSL));
    assert!(cell_is_vacuous(&cell(&[819]), OSL));
    assert!(!cell_is_vacuous(&cell(&[820]), OSL));
    assert!(!cell_is_vacuous(&cell(&[1024]), OSL));
}

/// What the rule now admits that the old one refused, pinned so the
/// admission is a documented decision rather than a discovery: at C=128 a
/// minority of empty requests passes up to the point where the cell's
/// total drops under 80%.
#[test]
fn the_admitted_worst_shapes_are_pinned() {
    let mut twenty_five_empty = vec![1024; 103];
    twenty_five_empty.extend([0; 25]);
    let d = Delivery::of(&cell(&twenty_five_empty), OSL);
    assert!(old_rule_voids(&twenty_five_empty, OSL));
    assert_eq!(d.cleared, 103);
    assert!(
        (d.delivered_pct() - 80.47).abs() < 0.01,
        "{}",
        d.delivered_pct()
    );
    assert!(
        !d.is_vacuous(),
        "25 empty of 128 is admitted: {}",
        d.describe(OSL)
    );

    let mut twenty_six_empty = vec![1024; 102];
    twenty_six_empty.extend([0; 26]);
    assert!(
        cell_is_vacuous(&cell(&twenty_six_empty), OSL),
        "26 empty of 128 is not"
    );

    // Half the batch stopping at 60% (615 of 1024): 80.02% delivered, 64 of
    // 128 cleared. Admitted.
    let mut half_at_60 = vec![1024; 64];
    half_at_60.extend([615; 64]);
    let d = Delivery::of(&cell(&half_at_60), OSL);
    assert_eq!(d.cleared, 64);
    assert!(d.delivered_pct() >= 80.0, "{}", d.delivered_pct());
    assert!(!d.is_vacuous(), "{}", d.describe(OSL));
    // One token less on each short request tips the total under the floor.
    let mut half_at_60_less = vec![1024; 64];
    half_at_60_less.extend([614; 64]);
    assert!(cell_is_vacuous(&cell(&half_at_60_less), OSL));
}

/// An empty cell is not vacuous — nothing to judge, and its error count
/// already voids it — and the warn text carries the numbers the rule used.
#[test]
fn empty_cells_and_the_evidence_line() {
    assert!(!cell_is_vacuous(&[], OSL));
    let d = Delivery::of(&cell(&[49]), 512);
    assert_eq!(
        d.describe(512),
        "delivered 9.6% of its 1×512-token budget and 0/1 request(s) cleared 80% of it (min 49 tok)"
    );
    assert_eq!(Delivery::of(&[], 512).delivered_pct(), 0.0);
}
