// SPDX-License-Identifier: AGPL-3.0-only

//! Provenance for a run whose serve config was overridden on the command line.
//!
//! Split from `tests.rs` for the 500-LoC cap.
//!
//! These exist because `--serve-override` breaks the assumption the rest of the
//! record format rests on: that `served_by` names a file you can open to see
//! what ran. Once an operator can change a recipe key at the command line, the
//! recipe id is a partial answer, and a partial answer that READS complete is
//! the failure mode this format was built against.

use super::tests::*;
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;

/// A run served with overrides says so in BOTH halves of its provenance.
///
/// ★ `served_by` alone would be a lie of omission here. A reader who opened
/// `qwen3.6-27b-nvfp4-unsloth.yaml` to see what produced these numbers would
/// find `kv_cache_dtype: bf16` and be reading the config that did NOT run —
/// the precise substitution the gate record format exists to make impossible.
/// So the overrides are a field of their own, and they also land in `command`
/// so the invocation still replays.
#[test]
fn a_run_with_serve_overrides_records_them_and_stays_replayable() {
    let mut overrides = BTreeMap::new();
    overrides.insert("kv_cache_dtype".to_string(), "fp8".to_string());
    overrides.insert("fp8_kv_calibration_tokens".to_string(), "512".to_string());
    let gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".to_string()),
        overrides.clone(),
    )
    .unwrap();
    assert_eq!(gate.serve_overrides, overrides);
    assert_eq!(
        gate.command,
        [
            "spark",
            "benchmark",
            "run",
            "bfcl-subset",
            "--param",
            "repeats=12",
            "--serve-override",
            "fp8_kv_calibration_tokens=512",
            "--serve-override",
            "kv_cache_dtype=fp8",
            "--pull-request-gate",
        ]
    );
}

/// The unmodified case stays clean: no field, no flag, and the JSON omits it.
///
/// A `"serve_overrides": {}` on every record would train readers to skip the
/// line that matters on the one record that has it.
#[test]
fn a_run_without_overrides_carries_no_override_provenance() {
    let gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".to_string()),
        Default::default(),
    )
    .unwrap();
    assert!(gate.serve_overrides.is_empty());
    assert!(!gate.command.join(" ").contains("--serve-override"));
    let json = serde_json::to_string(&gate).unwrap();
    assert!(!json.contains("serve_overrides"), "{json}");
}

/// A copied 16-slot BFCL record cannot cover a 256-slot pin.
///
/// BENCH.toml is outside the closure hash, so a pin-only edit would otherwise
/// leave an old record green. `check_record` demands the pin on the record.
#[test]
fn a_record_missing_a_baseline_serve_pin_fails() {
    let mut baseline = bfcl_baseline();
    baseline
        .hardware
        .get_mut(TEST_HW)
        .unwrap()
        .models
        .get_mut(MODEL)
        .unwrap()
        .serve_overrides
        .insert("ssm_cache_slots".into(), "256".into());

    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".into()),
        Default::default(),
    )
    .unwrap();
    let problems = check_record(&gate, &baseline).expect("must fail");
    assert_eq!(
        problems,
        [
            "serve override ssm_cache_slots=256 is pinned on the baseline but missing from the record"
        ]
    );
}

/// A record carrying the pin at the pinned value is judged on its metrics
/// exactly as before — the pin check adds no failure of its own.
#[test]
fn a_record_with_the_baseline_serve_pin_still_scores_metrics() {
    let mut baseline = bfcl_baseline();
    baseline
        .hardware
        .get_mut(TEST_HW)
        .unwrap()
        .models
        .get_mut(MODEL)
        .unwrap()
        .serve_overrides
        .insert("ssm_cache_slots".into(), "256".into());

    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let mut overrides = BTreeMap::new();
    overrides.insert("ssm_cache_slots".to_string(), "256".to_string());
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".into()),
        overrides,
    )
    .unwrap();
    assert!(check_record(&gate, &baseline).is_none());
}

#[test]
fn a_record_with_an_unpinned_serve_override_fails() {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        BTreeMap::from([("kv_cache_dtype".to_string(), "fp8".to_string())]),
    )
    .unwrap();
    let problems = check_record(&gate, &bfcl_baseline()).expect("must fail");
    assert_eq!(
        problems,
        [
            "serve override kv_cache_dtype=fp8 is present on the record but not pinned by the baseline"
        ]
    );
}

/// A record served at some OTHER value fails naming both numbers — the run
/// measured a config the baseline does not describe.
#[test]
fn a_record_with_a_different_pin_value_fails() {
    let mut baseline = bfcl_baseline();
    baseline
        .hardware
        .get_mut(TEST_HW)
        .unwrap()
        .models
        .get_mut(MODEL)
        .unwrap()
        .serve_overrides
        .insert("ssm_cache_slots".into(), "256".into());

    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let mut overrides = BTreeMap::new();
    overrides.insert("ssm_cache_slots".to_string(), "16".to_string());
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".into()),
        overrides,
    )
    .unwrap();
    let problems = check_record(&gate, &baseline).expect("must fail");
    assert_eq!(
        problems,
        ["serve override ssm_cache_slots=16 does not match the baseline pin ssm_cache_slots=256"]
    );
}

// ── Baseline PARAM pins on the record (param_overrides) ─────────────────────

fn baseline_with_param_pin(key: &str, value: &str) -> crate::gate::GateBaseline {
    let mut baseline = bfcl_baseline();
    baseline
        .hardware
        .get_mut(TEST_HW)
        .unwrap()
        .models
        .get_mut(MODEL)
        .unwrap()
        .param_overrides
        .insert(key.into(), value.into());
    baseline
}

/// A record measured WITHOUT a baseline-pinned parameter cannot cover the
/// pinned instrument. BENCH.toml is outside the closure hash, so pinning the
/// gate's ladder must not leave an old schema-default record reading green
/// for an instrument it never ran — the same argument as the serve pins.
#[test]
fn a_record_missing_a_baseline_param_pin_fails() {
    let baseline = baseline_with_param_pin("osl", "320");
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    let problems = check_record(&gate, &baseline).expect("must fail");
    assert_eq!(
        problems,
        ["param osl=320 is pinned on the baseline but missing from the record"]
    );
}

/// A record carrying the pin at the pinned value scores its metrics exactly
/// as before — and the comparison is whitespace-insensitive, because records
/// render int lists as "1, 4, 8, 16" while pins are typed "1,4,8,16".
#[test]
fn a_record_with_the_baseline_param_pin_scores_and_list_rendering_matches() {
    let baseline = baseline_with_param_pin("concurrencies", "1,4,8,16");
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let mut record = run_record(metrics, Verdict::pass("ok"));
    record
        .params
        .insert("concurrencies".into(), "1, 4, 8, 16".into());
    let gate = GateRecord::from_run(
        &record,
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    assert!(check_record(&gate, &baseline).is_none());
}

/// A record measured at some OTHER value fails naming both numbers — it ran
/// a different instrument than the one the thresholds describe.
#[test]
fn a_record_with_a_different_param_pin_value_fails() {
    let baseline = baseline_with_param_pin("osl", "320");
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let mut record = run_record(metrics, Verdict::pass("ok"));
    record.params.insert("osl".into(), "3 20".into());
    let gate = GateRecord::from_run(
        &record,
        hw(),
        SHA.into(),
        Vec::new(),
        None,
        Default::default(),
    )
    .unwrap();
    let problems = check_record(&gate, &baseline).expect("must fail");
    assert_eq!(
        problems,
        [
            "param osl=3 20 does not match the baseline pin osl=320 — the run measured a \
             different instrument than the one these thresholds describe"
        ]
    );
}
