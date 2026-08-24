// SPDX-License-Identifier: AGPL-3.0-only

//! Model-variant provenance: two checkpoints of one benchmark must never
//! overwrite, discharge, or be scored against each other.

use super::tests::{MODEL, SHA, TEST_HW, hw, run_record, tempdir, write_baseline};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;
use std::path::Path;

const DENSE: &str = "unsloth/Qwen3.8-27B-NVFP4";

/// A gb10 baseline where `bfcl-subset` is defined on the 35B (default) AND the
/// dense 27B, with different floors — the shape `agentic-webserver` now has in
/// the real tree.
fn two_variant_baseline() -> GateBaseline {
    let bound = |min: f64| Bound {
        min: Some(min),
        ..Bound::default()
    };
    let entry = |recipe: &str, min: f64| ModelBaseline {
        recipe: Some(recipe.to_string()),
        label: String::new(),
        note: String::new(),
        metrics: BTreeMap::from([("overall_accuracy".to_string(), bound(min))]),
        serve_overrides: BTreeMap::new(),
        param_overrides: BTreeMap::new(),
    };
    let mut models = BTreeMap::new();
    models.insert(MODEL.to_string(), entry("qwen3.6/moe", 84.0));
    models.insert(DENSE.to_string(), entry("qwen3.8/dense", 87.0));
    GateBaseline {
        schema: 2,
        hardware: BTreeMap::from([(
            TEST_HW.to_string(),
            HardwareBaseline {
                default: MODEL.to_string(),
                models,
            },
        )]),
    }
}

fn plant_variant(root: &Path, model: &str, secs: u64) -> std::path::PathBuf {
    let mut record = run_record(
        BTreeMap::from([("overall_accuracy".to_string(), 90.0)]),
        Verdict::pass("ok"),
    );
    record.target_model = model.to_string();
    record.recorded_at = secs;
    let gate = GateRecord::from_run(
        &record,
        hw(),
        SHA.to_string(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    write_record(root, &gate).unwrap()
}

#[test]
fn the_variant_slug_is_filename_safe_and_lossy_on_purpose() {
    assert_eq!(variant_slug(DENSE), "unsloth-qwen3.8-27b-nvfp4");
    assert_eq!(
        variant_slug("Qwen/Qwen3.6-35B-A3B-FP8"),
        "qwen-qwen3.6-35b-a3b-fp8"
    );
    assert_eq!(variant_slug("a//b"), "a-b", "no doubled separators");
    assert_eq!(variant_slug("/x/"), "x", "no leading or trailing separator");
}

/// One commit, one UTC day, both variants measured: the default keeps the
/// historical filename (nothing downstream moves), the non-default gets a
/// slugged one — and neither run REPLACES the other's record, which is the
/// silent-overwrite this naming exists to prevent.
#[test]
fn both_variants_records_coexist_for_one_commit_and_day() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    write_baseline(root, "bfcl-subset", &two_variant_baseline());

    let default_path = plant_variant(root, MODEL, 1_785_891_382);
    let dense_path = plant_variant(root, DENSE, 1_785_891_382 + 60);

    assert_eq!(
        default_path,
        root.join(".benchmarks/bfcl-subset/2026-08-05-b72dad1893.json")
    );
    assert_eq!(
        dense_path,
        root.join(".benchmarks/bfcl-subset/2026-08-05-b72dad1893-unsloth-qwen3.8-27b-nvfp4.json")
    );
    assert!(default_path.exists() && dense_path.exists());
    // Each file still says which variant produced it, independent of its name.
    assert_eq!(read_record(&default_path).unwrap().target_model, MODEL);
    assert_eq!(read_record(&dense_path).unwrap().target_model, DENSE);
}

#[test]
fn lossy_variant_slugs_cannot_overwrite_each_other() {
    const ALIAS: &str = "unsloth/Qwen3.8/27B/NVFP4";
    assert_eq!(variant_slug(DENSE), variant_slug(ALIAS));
    let dir = tempdir::Dir::new();
    let root = dir.path();
    let mut baseline = two_variant_baseline();
    let dense = baseline.hardware[TEST_HW].models[DENSE].clone();
    baseline
        .hardware
        .get_mut(TEST_HW)
        .unwrap()
        .models
        .insert(ALIAS.into(), dense);
    write_baseline(root, "bfcl-subset", &baseline);

    let first = plant_variant(root, DENSE, 1_785_891_382);
    let second = plant_variant(root, ALIAS, 1_785_891_382 + 60);
    assert_ne!(first, second);
    assert_eq!(read_record(&first).unwrap().target_model, DENSE);
    assert_eq!(read_record(&second).unwrap().target_model, ALIAS);
}

/// A same-day re-run of the SAME non-default variant still replaces its own
/// record — the per-variant file is the branch's current word on that variant.
#[test]
fn a_variant_rerun_replaces_only_its_own_record() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    write_baseline(root, "bfcl-subset", &two_variant_baseline());
    let first = plant_variant(root, DENSE, 1_785_891_382);
    let second = plant_variant(root, DENSE, 1_785_891_382 + 3_600);
    assert_eq!(first, second, "same variant + sha + UTC day = same file");
    assert_eq!(read_record(&second).unwrap().recorded_at, 1_785_894_982);
}

/// A benchmark with no baseline at all keeps the legacy name: there is no
/// variant axis declared, so there is nothing to key by.
#[test]
fn no_baseline_means_the_legacy_filename() {
    let dir = tempdir::Dir::new();
    let path = plant_variant(dir.path(), DENSE, 1_785_891_382);
    assert_eq!(
        path,
        dir.path()
            .join(".benchmarks/bfcl-subset/2026-08-05-b72dad1893.json")
    );
}

/// ★ A degraded hardware fingerprint must not reopen the overwrite hole.
/// `fetch_hardware` degrades to `Hardware::unknown()` without surfacing an
/// error; before the fix, an unknown box class fell through to the legacy
/// filename even for a NON-default variant, so a dense run with a failed
/// probe silently destroyed the committed default record for the same
/// commit and day. Non-default → slugged, regardless of hardware; a model
/// that IS a declared default keeps the legacy name (historical behavior).
#[test]
fn an_unknown_hardware_variant_record_does_not_clobber_the_default() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    write_baseline(root, "bfcl-subset", &two_variant_baseline());
    let default_path = plant_variant(root, MODEL, 1_785_891_382);

    let mut record = run_record(
        BTreeMap::from([("overall_accuracy".to_string(), 90.0)]),
        Verdict::pass("ok"),
    );
    record.target_model = DENSE.to_string();
    record.recorded_at = 1_785_891_382 + 60;
    let gate = GateRecord::from_run(
        &record,
        crate::hardware::Hardware::unknown(),
        SHA.to_string(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    let dense_path = write_record(root, &gate).unwrap();

    assert_eq!(
        dense_path,
        root.join(".benchmarks/bfcl-subset/2026-08-05-b72dad1893-unsloth-qwen3.8-27b-nvfp4.json")
    );
    // The default's committed record survives untouched.
    assert_eq!(read_record(&default_path).unwrap().target_model, MODEL);

    // And an unknown-hardware record of the DEFAULT model keeps the legacy
    // name — the pre-variant era behavior this naming promised not to move.
    let mut record = run_record(
        BTreeMap::from([("overall_accuracy".to_string(), 90.0)]),
        Verdict::pass("ok"),
    );
    record.target_model = MODEL.to_string();
    record.recorded_at = 1_785_891_382 + 120;
    let gate = GateRecord::from_run(
        &record,
        crate::hardware::Hardware::unknown(),
        SHA.to_string(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    assert_eq!(write_record(root, &gate).unwrap(), default_path);
}

/// ★ The required gate's subject is the DEFAULT checkpoint. A newer, passing
/// record of another variant must not become "the branch's current word" on
/// it — a plausible green attached to the wrong subject is the single worst
/// outcome of the variant feature.
#[test]
fn a_non_default_variant_record_cannot_discharge_the_required_gate() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    for id in REQUIRED_GATES {
        write_baseline(root, id, &two_variant_baseline());
    }
    // The dense record is NEWER than the default's — the ordering that used to
    // let it shadow the required subject.
    plant_variant(root, MODEL, 1_785_891_382);
    plant_variant(root, DENSE, 1_785_891_382 + 60);

    let gates = check_gates(root, SHA);
    assert!(
        matches!(gates["bfcl-subset"], GateStatus::Pass),
        "the OLDER default record still discharges the gate: {:?}",
        gates["bfcl-subset"]
    );

    // Remove the default's record: the dense one alone must read as MISSING,
    // never as a pass for the 35B subject.
    std::fs::remove_file(record_path(root, "bfcl-subset", 1_785_891_382, SHA)).unwrap();
    let gates = check_gates(root, SHA);
    assert!(matches!(
        &gates["bfcl-subset"],
        GateStatus::Missing(reason)
            if reason == "latest record measured the unsloth/Qwen3.8-27B-NVFP4 variant; the required subject on gb10 is Qwen/Qwen3.6-35B-A3B-FP8, which has no covering record"
    ));
}

/// The variant record is still SCORED against its own thresholds when asked
/// directly — exclusion from the required gate is about subject, not validity.
#[test]
fn a_variant_record_is_scored_against_its_own_entry() {
    let baseline = two_variant_baseline();
    let mut record = run_record(
        BTreeMap::from([("overall_accuracy".to_string(), 86.0)]),
        Verdict::pass("ok"),
    );
    record.target_model = DENSE.to_string();
    let gate = GateRecord::from_run(
        &record,
        hw(),
        SHA.to_string(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    // 86.0 clears the MoE floor (84.0) but not the dense one (87.0): only a
    // dense-floor failure proves the dense entry is what it was scored on.
    assert_eq!(
        check_record(&gate, &baseline),
        Some(vec![
            "overall_accuracy 86.00 is below the floor 87.00 (noise 0.00)".into()
        ])
    );
}
