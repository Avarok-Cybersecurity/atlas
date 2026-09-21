// SPDX-License-Identifier: AGPL-3.0-only

//! The concurrency sweep's RUN-VERDICT tests: the floors the gate fills, the
//! rules those floors can never override (request errors, vacuous cells, an
//! uncontrolled cache), and the committed ladder they are read from.
//!
//! Split from `concurrency_tests.rs` for the 500-LoC cap. Exact piecewise
//! copy — no test changed in the move.
//!
//! The essay-fixture tests moved to `concurrency_essay_tests.rs` when the
//! ratchet rationale tipped this file over the cap. They are a CHILD module
//! of this one, not a sibling: a sibling would need its own `gate::coverage`
//! registry row, which re-opens every gate; a child needs none.

use super::verdict::{Exclusions, Floors, sweep_verdict};
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
    // The 2026-08-30 re-cut, from 3 reps of the WIDENED C=1..128 instrument.
    // The rung set is read from the committed file rather than re-typed, so
    // adding a rung moves this assertion instead of failing it; the VALUES are
    // pinned because a silent floor change is the thing worth catching.
    //
    // ★ REPINNED 2026-09-21 to the PROVISIONAL ladder. The gate moved to the
    // published instrument (ISL 128 / OSL 1024 / essay); the old bars were cut
    // at 512/320 and describe nothing here, so each is the published Atlas
    // value at that rung x 0.5, rounded down. The fixture below is the
    // published ladder itself, which is what this gate should now measure.
    //
    // ★ ZERO WAS NOT AN OPTION, and this pin is where that is remembered.
    // `RUNGS` says a rung with no metrics block gates nothing — true of ONE
    // rung beside bounded ones. But `Floors::gating()` is
    // `peak > 0 || any per_c > 0`, so ALL floors at zero flips the run to the
    // INFO verdict, and `verdict_passes` accepts only "PASS": an all-zero
    // ladder makes this gate unsatisfiable rather than ungated. The assertion
    // below therefore pins that every floor is > 0 as well as its value.
    let m = ladder(&[
        ("c1_aggregate_tok_s", 23.59),
        ("c2_aggregate_tok_s", 41.02),
        ("c4_aggregate_tok_s", 74.21),
        ("c8_aggregate_tok_s", 125.95),
        ("c16_aggregate_tok_s", 203.36),
        ("c32_aggregate_tok_s", 291.01),
        ("c64_aggregate_tok_s", 386.63),
        ("c128_aggregate_tok_s", 478.11),
        ("peak_aggregate_tok_s", 478.11),
    ]);
    let floors = committed_floors();
    assert_eq!(
        floors.per_c,
        vec![
            (1, 11.5),
            (2, 20.5),
            (4, 37.0),
            (8, 62.5),
            (16, 100.0),
            (32, 140.0),
            (64, 190.0),
            (128, 230.0)
        ]
    );
    assert_eq!(floors.peak, 230.0);
    assert!(
        floors.gating(),
        "an all-zero ladder is an INFO verdict, not an ungated one, and INFO is \
         not a PASS — the gate would be unsatisfiable"
    );
    let v = sweep_verdict(
        &m,
        8,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &floors,
    );
    assert_eq!(v.kind, VerdictKind::Pass, "{}", v.reason);
    for rung in ["C1", "C2", "C4", "C8", "C16", "C32", "C64", "C128", "peak"] {
        assert!(v.reason.contains(rung), "{}", v.reason);
    }

}

/// FAIL names the violating cell — and the comparison is the raw value
/// against the floor, deliberately stricter than gate scoring's
/// value + noise >= min.
#[test]
fn a_sweep_below_one_floor_fails_naming_the_cell() {
    let committed = committed_floors();
    // C=8 is found by VALUE rather than by position so a rung inserted ahead
    // of it cannot silently retarget this test at a different cell.
    let c8_floor = committed
        .per_c
        .iter()
        .find(|(c, _)| *c == 8)
        .expect("C=8 is a committed rung")
        .1;
    // ★ REBASED 2026-09-21 onto the published ladder, the instrument this gate
    // now measures; the old 21.2/27.7/44.8 fixture is a 512/320 reading and
    // would fail five OTHER rungs against the current bars, which would leave
    // this test passing for the wrong reason. Every rung is the published
    // Atlas value except C=8, which is derived a hair under its own committed
    // floor rather than typed — so a re-cut moves the fixture with the bar
    // instead of silently making this test measure nothing.
    let c8_under = c8_floor - 0.1;
    let m = ladder(&[
        ("c1_aggregate_tok_s", 23.59),
        ("c2_aggregate_tok_s", 41.02),
        ("c4_aggregate_tok_s", 74.21),
        ("c8_aggregate_tok_s", c8_under),
        ("c16_aggregate_tok_s", 203.36),
        ("c32_aggregate_tok_s", 291.01),
        ("c64_aggregate_tok_s", 386.63),
        ("c128_aggregate_tok_s", 478.11),
        ("peak_aggregate_tok_s", 478.11),
    ]);
    let v = sweep_verdict(
        &m,
        8,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &committed,
    );
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("C=8"), "{}", v.reason);
    assert!(
        v.reason.contains(&format!("{c8_under:.1}")) && v.reason.contains(&format!("{c8_floor:.1}")),
        "{}",
        v.reason
    );
    // ...and ONLY C=8: the other eight rungs clear their bars, so this test
    // fails for the reason it names rather than for five of them at once.
    for other in ["C=1", "C=2", "C=4", "C=16", "C=32", "C=64", "C=128"] {
        assert!(
            !v.reason.contains(&format!("{other} ")),
            "{other} should clear its floor: {}",
            v.reason
        );
    }
    // Exactly on the floor passes — inclusive, like the BENCH.toml bound.
    let m = ladder(&[("c8_aggregate_tok_s", c8_floor)]);
    let v = sweep_verdict(
        &m,
        1,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &floors(0.0, 0.0, c8_floor, 0.0, 0.0),
    );
    assert_eq!(v.kind, VerdictKind::Pass, "{}", v.reason);
}

#[test]
fn all_floors_zero_keeps_the_info_verdicts() {
    let m = ladder(&[("c1_aggregate_tok_s", 25.5)]);
    let clean = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &Floors::default(),
    );
    assert_eq!(clean.kind, VerdictKind::Info, "{}", clean.reason);
    assert!(
        clean.reason.contains("no request errors"),
        "{}",
        clean.reason
    );
    let vac = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 2,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &Floors::default(),
    );
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
    let v = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 1,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &floors(24.0, 43.0, 63.0, 94.0, 94.0),
    );
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
    let v = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &floors(0.0, 0.0, 0.0, 94.0, 0.0),
    );
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("C=16"), "{}", v.reason);
    assert!(v.reason.contains("INCONCLUSIVE"), "{}", v.reason);
    // Same for the peak floor.
    let v = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &floors(0.0, 0.0, 0.0, 0.0, 94.0),
    );
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("peak"), "{}", v.reason);
}

#[test]
fn request_errors_fail_the_sweep_in_both_modes() {
    let m = ladder(&[("c1_aggregate_tok_s", 999.0)]);
    for f in [Floors::default(), floors(24.0, 43.0, 63.0, 94.0, 94.0)] {
        let v = sweep_verdict(
            &m,
            4,
            2,
            Exclusions {
                vacuous: 0,
                cache_uncontrolled: 0,
                non_mtp_arm: 0,
            },
            80.0,
            &f,
        );
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
    let v = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 1,
            non_mtp_arm: 0,
        },
        80.0,
        &floors(24.0, 43.0, 63.0, 94.0, 94.0),
    );

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

#[path = "concurrency_essay_tests.rs"]
mod essay;
