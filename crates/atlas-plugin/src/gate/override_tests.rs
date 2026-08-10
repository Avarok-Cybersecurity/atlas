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
    let joined = gate.command.join(" ");
    assert!(
        joined.contains("--serve-override kv_cache_dtype=fp8"),
        "the command must replay the override, not just the recipe: {joined}"
    );
    assert!(
        joined.contains("--serve-override fp8_kv_calibration_tokens=512"),
        "{joined}"
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
