// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn repo_root() -> std::path::PathBuf {
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

fn fixture(name: &str, bench_toml: &str) -> std::path::PathBuf {
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

/// A `[benchmarks.serve_overrides]` table is the SSOT for a gate-local pin.
#[test]
fn serve_overrides_are_assembled_into_the_baseline() {
    let root = fixture(
        "serve-overrides",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.serve_overrides]
ssm_cache_slots = "256"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    let baseline = baseline_for(&root, "bfcl-subset").unwrap();
    let (checkpoint, entry) = baseline.resolve("gb10", None).unwrap();
    assert_eq!(checkpoint, "org/A");
    assert_eq!(
        entry.serve_overrides,
        std::collections::BTreeMap::from([("ssm_cache_slots".to_string(), "256".to_string())])
    );
}

/// `port` is owned by self-start. A pin here would name a listener that is not
/// there, so it is refused at parse rather than dropped later.
#[test]
fn a_port_serve_override_is_refused() {
    let root = fixture(
        "port-pin",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.serve_overrides]
port = "8888"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    let err = load_all(&root).unwrap_err().to_string();
    assert_eq!(
        err,
        format!(
            "{}: bfcl-subset / org/A serve_overrides cannot set `port`: self-start binds a free port itself, so a pin here would name a listener that is not there",
            root.join("kernels/gb10/modelA/BENCH.toml").display()
        )
    );
}

/// The committed tree's pins, exactly where the gates need them — and nowhere
/// else. The echolp pin must not move the floors: those are the high-water
/// ratchet, and this change is capacity (Marconi pool), not a score lever.
#[test]
fn the_trees_serve_pins_sit_on_the_gates_that_need_them() {
    let root = repo_root();

    // Gate B: the 35B echolp draw self-starts with the Marconi pool pinned, so
    // a 1004-sample serial generate cannot evict its own snapshots — with the
    // ratcheted floors untouched.
    let echolp = baseline_for(&root, "bfcl-subset-echolp").unwrap();
    let (_, e) = echolp.resolve("gb10", None).unwrap();
    assert_eq!(e.metrics["overall_accuracy"].min, Some(86.50));
    assert_eq!(e.metrics["normalized_single_turn_score"].min, Some(86.90));
    assert_eq!(e.metrics["samples"].min, Some(1004.0));
    assert_eq!(e.metrics["samples"].max, Some(1004.0));
    assert_eq!(
        e.serve_overrides.get("ssm_cache_slots").map(String::as_str),
        Some("256")
    );
    assert_eq!(e.serve_overrides.len(), 1, "{:?}", e.serve_overrides);

    // The poison gate declares BOTH of its documented serve deltas, so a
    // `--pull-request-gate` run needs no operator flags at all and still
    // matches the config probe.rs documents as required.
    let poison = baseline_for(&root, "ssm-state-poisoning-gate").unwrap();
    let (_, p) = poison.resolve("gb10", None).unwrap();
    assert_eq!(
        p.serve_overrides.get("ssm_cache_slots").map(String::as_str),
        Some("256")
    );
    assert_eq!(
        p.serve_overrides
            .get("disable_thinking")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(p.serve_overrides.len(), 2, "{:?}", p.serve_overrides);

    // The concurrency gate declares its whole batched serve profile: the
    // shared agentic recipe is a serial reproduction config (batch 1, bf16 KV,
    // 256 Marconi slots, 32K context) that strangles a concurrency instrument.
    // lm_head_dtype is deliberately absent — the recipe's bf16 head is a
    // correctness pin, not a throughput knob. Marconi is pinned at 8 slots
    // (2026-08-16): concurrency serving does not replay long shared prefixes,
    // so 8 retains the warm-prefix value at 1/4 the 32-slot reserve
    // (151.5 MiB/slot — see the BENCH.toml comment for the byte math).
    let sweep = baseline_for(&root, "concurrency-sweep").unwrap();
    let (_, c) = sweep.resolve("gb10", None).unwrap();
    for (key, want) in [
        ("max_batch_size", "32"),
        ("kv_cache_dtype", "fp8"),
        ("ssm_cache_slots", "8"),
        ("max_model_len", "4096"),
    ] {
        assert_eq!(
            c.serve_overrides.get(key).map(String::as_str),
            Some(want),
            "concurrency-sweep serve pin {key}: {:?}",
            c.serve_overrides
        );
    }
    assert_eq!(c.serve_overrides.len(), 4, "{:?}", c.serve_overrides);
    assert!(
        !c.serve_overrides.contains_key("lm_head_dtype"),
        "the bf16 head is a correctness pin the gate must not touch"
    );

    // Everything else keeps the recipe's own config. bfcl-subset in
    // particular: its default subject's bars (Qwen3.8-27B, 2026-08-14) were
    // measured WITHOUT a pin, and a pin added after the fact would desync the
    // thresholds from the config that produced them.
    //
    // ★ decode-floor is in this list as a REGRESSION PIN: on 2026-08-15 the
    // concurrency gate's serve pin was committed ABOVE its own [[benchmarks]]
    // header, so TOML attached it to the PRECEDING decode-floor entry — the
    // concurrency gate then served the recipe's batch 1 (silently: the
    // OVERRIDES disclosure only prints for a non-empty merged set) while the
    // NEXT decode-floor run would have served batch 32, a different
    // instrument than its floor describes.
    for id in [
        "bfcl-subset",
        "ttft-warm-gate",
        "ttft-cold-gate",
        "agentic-webserver",
        "decode-floor",
    ] {
        let b = baseline_for(&root, id).unwrap();
        let (_, entry) = b.resolve("gb10", None).unwrap();
        assert!(
            entry.serve_overrides.is_empty(),
            "{id} keeps the recipe's own config: {:?}",
            entry.serve_overrides
        );
    }
}

#[path = "bench_override_tree_tests.rs"]
mod bench_override_tree_tests;
