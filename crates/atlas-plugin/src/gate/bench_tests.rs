// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

pub(super) fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("atlas-bench-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

pub(super) fn fixture(name: &str, bench_toml: &str) -> std::path::PathBuf {
    let root = tmp(name);
    let m = root.join("kernels/gb10/modelA");
    std::fs::create_dir_all(m.join("nvfp4")).unwrap();
    std::fs::write(
        root.join("kernels/gb10/HARDWARE.toml"),
        "[hardware]\nvendor = \"nvidia\"\n",
    )
    .unwrap();
    std::fs::write(m.join("MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(m.join("BENCH.toml"), bench_toml).unwrap();
    root
}

// ---------------------------------------------------------------------------
// The tree the gate actually reads
// ---------------------------------------------------------------------------

// The migration from `.benchmarks/<gate>/BASELINE.json` was proven by an
// equivalence test that asserted every assembled baseline matched the JSON it
// replaced, field for field, across all five gates. It is deleted along with
// the JSON: kept, it would pin BENCH.toml to a frozen snapshot and fail on the
// first legitimate ratchet. The proof is in the commit that performed the move.

/// Every gate the checker requires must resolve a baseline from the tree, or
/// the migration silently un-gates something.
#[test]
fn every_required_gate_still_resolves_a_baseline() {
    let root = repo_root();
    for id in super::super::REQUIRED_GATES {
        let baseline = baseline_for(&root, id).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert!(
            !baseline.hardware.is_empty(),
            "{id}: no hardware entries after the move"
        );
        for (hw, entry) in &baseline.hardware {
            assert!(
                !entry.default.is_empty(),
                "{id}/{hw}: no default checkpoint"
            );
            assert!(
                entry.models.contains_key(&entry.default),
                "{id}/{hw}: default {:?} is not among its own models",
                entry.default
            );
        }
    }
}

/// The thresholds must not be reachable from a `kernels/` edit's blast radius
/// in the wrong direction: `BENCH.toml` is deliberately not a closure input, so
/// ratcheting a bar does not invalidate the record that justified it.
#[test]
fn bench_toml_is_not_a_closure_input() {
    let root = repo_root();
    let target = taxon::Target {
        hardware: "gb10".into(),
        model: "qwen3.6-27b".into(),
        quant: "nvfp4".into(),
    };
    let configs = taxon::configs(&root, &target);
    assert!(
        !configs.iter().any(|p| p.ends_with("BENCH.toml")),
        "BENCH.toml must not be hashed: a threshold ratchet would invalidate \
         the very record that justified it. Found: {configs:?}"
    );
    assert!(
        configs.iter().any(|p| p.ends_with("MODEL.toml")),
        "MODEL.toml IS compiled in and must stay hashed: {configs:?}"
    );
}

// ---------------------------------------------------------------------------
// Schema rules
// ---------------------------------------------------------------------------

/// ★ A guess can never go green. An unmeasured entry with thresholds is
/// rejected outright rather than trusted.
#[test]
fn an_unmeasured_entry_carrying_thresholds_is_rejected() {
    let root = fixture(
        "guessed",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "unmeasured"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    let err = load_all(&root).unwrap_err().to_string();
    assert_eq!(
        err,
        format!(
            "{}: bfcl-subset / org/M is unmeasured but carries thresholds. A guessed number a run can clear is worse than no number — it reports PASS for something nobody measured.",
            root.join("kernels/gb10/modelA/BENCH.toml").display()
        )
    );
}

/// The mirror: claiming `measured` without numbers is equally a lie.
#[test]
fn a_measured_entry_without_thresholds_is_rejected() {
    for (name, metrics) in [("absent", ""), ("empty", "[benchmarks.metrics]")] {
        let root = fixture(
            name,
            &format!(
                r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "measured"
{metrics}
"#
            ),
        );
        let err = load_all(&root).unwrap_err().to_string();
        assert_eq!(
            err,
            format!(
                "{}: bfcl-subset / org/M claims to be measured but declares no metrics",
                root.join("kernels/gb10/modelA/BENCH.toml").display()
            ),
            "partition {name}"
        );
    }
}

#[test]
fn an_unknown_status_is_rejected_rather_than_treated_as_unmeasured() {
    let root = fixture(
        "bad-status",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "probably-fine"
"#,
    );
    assert_eq!(
        load_all(&root).unwrap_err().to_string(),
        format!(
            "{}: status must be \"measured\" or \"unmeasured\", got \"probably-fine\"",
            root.join("kernels/gb10/modelA/BENCH.toml").display()
        )
    );
}

/// An unmeasured entry is DROPPED from the baseline, so the gate reports
/// "no baseline for model X" rather than passing everything it is given.
#[test]
fn an_unmeasured_entry_produces_no_baseline_at_all() {
    let root = fixture(
        "unmeasured",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "unmeasured"
"#,
    );
    // The baseline assembles EMPTY rather than erroring — there is no hardware
    // to check a default on. What matters is the next step: resolving against
    // it must fail loudly, never read as "nothing to check".
    let baseline = baseline_for(&root, "bfcl-subset").unwrap();
    assert_eq!(baseline.schema, 2);
    assert!(baseline.hardware.is_empty(), "{baseline:?}");
    let err = baseline.resolve("gb10", None).unwrap_err().to_string();
    assert_eq!(
        err,
        "no baseline for hardware \"gb10\"; this benchmark has entries for []"
    );
}

/// ★ Two defaults is a coin-flip over which checkpoint a gate scores, and two
/// checkpoints of one model can differ by several BFCL points.
#[test]
fn two_checkpoints_claiming_default_is_an_error() {
    let root = fixture(
        "two-defaults",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0

[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/B"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 86.0
"#,
    );
    let err = baseline_for(&root, "bfcl-subset").unwrap_err().to_string();
    assert_eq!(
        err,
        "bfcl-subset: both org/A (in modelA) and org/B (in modelA) claim to be the default on gb10"
    );
}

/// No implicit "the only entry wins": a second checkpoint added later would
/// silently move which one the gate scores.
#[test]
fn a_lone_checkpoint_must_still_declare_itself_default() {
    let root = fixture(
        "no-default",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    assert_eq!(
        baseline_for(&root, "bfcl-subset").unwrap_err().to_string(),
        "bfcl-subset: no checkpoint on gb10 sets `default = true`; one must, or the gate has no defined subject"
    );
}

#[test]
fn the_same_checkpoint_declared_twice_for_one_gate_is_an_error() {
    let root = fixture(
        "dupe",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0

[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 90.0
"#,
    );
    assert_eq!(
        baseline_for(&root, "bfcl-subset").unwrap_err().to_string(),
        "bfcl-subset: org/A is declared twice on gb10"
    );
}

/// A model with several quant dirs must not have its entries counted once per
/// quant — `walk` yields one target per quant, but the file is per model.
#[test]
fn entries_are_not_duplicated_across_a_models_quant_dirs() {
    let root = fixture(
        "multi-quant",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    std::fs::create_dir_all(root.join("kernels/gb10/modelA/fp8")).unwrap();
    let all = load_all(&root).unwrap();
    assert_eq!(all.len(), 1, "one entry, not one per quant dir: {all:?}");
}

// ---------------------------------------------------------------------------
// Baseline-declared serve pins ([benchmarks.serve_overrides])
// ---------------------------------------------------------------------------
