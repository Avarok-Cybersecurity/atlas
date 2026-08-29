// SPDX-License-Identifier: AGPL-3.0-only

//! The concurrency sweep's RUN-VERDICT tests: the floors the gate fills, the
//! rules those floors can never override (request errors, vacuous cells, an
//! uncontrolled cache), and the committed ladder they are read from.
//!
//! Split from `concurrency_tests.rs` for the 500-LoC cap. Exact piecewise
//! copy — no test changed in the move.

use super::verdict::{Floors, sweep_verdict};
use super::*;
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
    // Only the rungs that carry a committed bound. A rung the ladder measures
    // but nobody has bounded yet is absent from BENCH.toml on purpose (see
    // RUNGS in concurrency.rs) and must not be invented here as 0.0, or this
    // helper would claim a floor exists where none was measured.
    let floor = |metric: &str| metrics.get(metric).and_then(|b| b.min);
    Floors {
        per_c: [1usize, 2, 4, 8, 16, 32, 64, 128]
            .into_iter()
            .filter_map(|c| floor(&format!("c{c}_aggregate_tok_s")).map(|v| (c, v)))
            .collect(),
        peak: min("peak_aggregate_tok_s"),
    }
}

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
    let v = sweep_verdict(&m, 4, 0, 0, 0, 80.0, &floors);
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
    let v = sweep_verdict(&m, 4, 0, 0, 0, 80.0, &committed);
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("C=8"), "{}", v.reason);
    assert!(
        v.reason.contains("51.2") && v.reason.contains("52.0"),
        "{}",
        v.reason
    );
    // Exactly on the floor passes — inclusive, like the BENCH.toml bound.
    let m = ladder(&[("c8_aggregate_tok_s", c8_floor)]);
    let v = sweep_verdict(&m, 1, 0, 0, 0, 80.0, &floors(0.0, 0.0, c8_floor, 0.0, 0.0));
    assert_eq!(v.kind, VerdictKind::Pass, "{}", v.reason);
}

#[test]
fn all_floors_zero_keeps_the_info_verdicts() {
    let m = ladder(&[("c1_aggregate_tok_s", 25.5)]);
    let clean = sweep_verdict(&m, 4, 0, 0, 0, 80.0, &Floors::default());
    assert_eq!(clean.kind, VerdictKind::Info, "{}", clean.reason);
    assert!(
        clean.reason.contains("no request errors"),
        "{}",
        clean.reason
    );
    let vac = sweep_verdict(&m, 4, 0, 2, 0, 80.0, &Floors::default());
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
    let v = sweep_verdict(&m, 4, 0, 1, 0, 80.0, &floors(24.0, 43.0, 63.0, 94.0, 94.0));
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
    let v = sweep_verdict(&m, 4, 0, 0, 0, 80.0, &floors(0.0, 0.0, 0.0, 94.0, 0.0));
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("C=16"), "{}", v.reason);
    assert!(v.reason.contains("INCONCLUSIVE"), "{}", v.reason);
    // Same for the peak floor.
    let v = sweep_verdict(&m, 4, 0, 0, 0, 80.0, &floors(0.0, 0.0, 0.0, 0.0, 94.0));
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("peak"), "{}", v.reason);
}

#[test]
fn request_errors_fail_the_sweep_in_both_modes() {
    let m = ladder(&[("c1_aggregate_tok_s", 999.0)]);
    for f in [Floors::default(), floors(24.0, 43.0, 63.0, 94.0, 94.0)] {
        let v = sweep_verdict(&m, 4, 2, 0, 0, 80.0, &f);
        assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
        assert!(v.reason.contains("2 request(s) failed"), "{}", v.reason);
    }
}

#[test]
fn an_unobserved_warm_cache_cannot_clear_the_gate() {
    let m = ladder(&[
        ("c1_aggregate_tok_s", 999.0),
        ("c4_aggregate_tok_s", 999.0),
        ("c8_aggregate_tok_s", 999.0),
        ("c16_aggregate_tok_s", 999.0),
        ("peak_aggregate_tok_s", 999.0),
    ]);
    let v = sweep_verdict(&m, 4, 0, 0, 1, 80.0, &floors(24.0, 43.0, 63.0, 94.0, 94.0));

    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("cached-prompt fraction"), "{}", v.reason);
}

/// The descriptor couples each floor param to the metric its BENCH.toml bound
/// is written on, every param exists in the schema with the documented OFF
/// default, and `configure` carries the values into the verdict floors.
/// Every RUNGS row must spell its own C. A row like
/// `(32, "min_c32", "c64_aggregate_tok_s", ..)` is silent in both directions:
/// `apply_threshold_params` would fill the C=32 floor from the C=64 bound, and
/// `sweep_verdict` would then judge C=32's throughput against it. Nothing
/// downstream can notice, because both names are well-formed.
#[test]
fn every_rung_names_itself_consistently() {
    for (c, param, metric, label) in RUNGS {
        assert_eq!(param, format!("min_c{c}"), "rung {c}: floor param");
        assert_eq!(
            metric,
            format!("c{c}_aggregate_tok_s"),
            "rung {c}: metric key"
        );
        assert!(
            label.contains(&format!("C={c} ")),
            "rung {c}: label {label:?}"
        );
    }
    let mut seen: Vec<usize> = RUNGS.iter().map(|(c, ..)| *c).collect();
    let sorted = {
        let mut v = seen.clone();
        v.sort_unstable();
        v
    };
    assert_eq!(seen, sorted, "rungs must be in ascending order");
    seen.dedup();
    assert_eq!(seen.len(), RUNGS.len(), "a rung is declared twice");
    // The peak is not a rung and must never be mistaken for one: a
    // `c<N>_aggregate_tok_s` key here would make the peak floor gate a rung.
    assert_eq!(PEAK_FLOOR.1, "peak_aggregate_tok_s");
    assert!(
        RUNGS
            .iter()
            .all(|(_, p, m, _)| *p != PEAK_FLOOR.0 && *m != PEAK_FLOOR.1)
    );
}

#[test]
fn the_floor_params_are_wired_to_the_gate() {
    // Derived from RUNGS, so this asserts the DERIVATION rather than
    // re-typing the list: a rung added there must appear here, paired with
    // the metric key the BENCH.toml bound is written on, and the peak must
    // stay last.
    assert_eq!(
        DESCRIPTOR.threshold_params,
        [
            ("min_c1", "c1_aggregate_tok_s"),
            ("min_c2", "c2_aggregate_tok_s"),
            ("min_c4", "c4_aggregate_tok_s"),
            ("min_c8", "c8_aggregate_tok_s"),
            ("min_c16", "c16_aggregate_tok_s"),
            ("min_c32", "c32_aggregate_tok_s"),
            ("min_c64", "c64_aggregate_tok_s"),
            ("min_c128", "c128_aggregate_tok_s"),
            ("min_peak", "peak_aggregate_tok_s"),
        ]
    );
    // Both gate ids share the driver, so they must also share the pairing —
    // a DFlash2 record scored against a different set of metric names would
    // silently gate nothing.
    assert_eq!(
        DFLASH2_DESCRIPTOR.threshold_params, DESCRIPTOR.threshold_params,
        "the two concurrency gates must gate on the same metric names"
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
        vec![
            (1, 0.0),
            (2, 0.0),
            (4, 0.0),
            (8, 63.0),
            (16, 0.0),
            (32, 0.0),
            (64, 0.0),
            (128, 0.0)
        ],
        "an unbounded rung must carry the 0.0 OFF value, not be absent — \
         `sweep_verdict` fails a GATED rung that produced no comparable cell"
    );
    assert_eq!(b.floors.peak, 94.0);
}
