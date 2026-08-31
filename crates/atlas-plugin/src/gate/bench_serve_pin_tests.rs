// SPDX-License-Identifier: AGPL-3.0-only

//! What a gate's SERVE PIN must be, and what it must not.
//!
//! `[benchmarks.serve_overrides]` is the half of a BENCH.toml entry that
//! decides what the gate actually launches, and `check_record` demands an
//! exact bidirectional match afterwards — so a pin that drifts silently
//! invalidates records rather than changing a number. These tests pin where
//! each committed override sits and which gate owns it.
//!
//! Split from `bench_tests.rs` for the 500-LoC cap when the concurrency gates
//! grew a second subject. Exact piecewise copy — no test changed in the move.

use super::bench_tests::{fixture, repo_root};
use super::*;

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
        // 128, with the ladder: a batch cap below the widest measured rung
        // makes that rung serial, which is the recipe's batch-1 defect one
        // scale up.
        ("max_batch_size", "128"),
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

    // The DFlash2 gate is the same profile PLUS the drafter, and nothing else.
    // Each of the three drafter keys is load-bearing: without `dflash` the run
    // measures the base engine under a speculative label, without an explicit
    // `draft_model` it depends on a MODEL.toml fallback the record would not
    // disclose, and without `dflash_gamma` the CLI default of 16 shadows this
    // drafter's trained block size of 8 — which measured 0% accept on every
    // verify step.
    let dflash2 = baseline_for(&root, "concurrency-sweep-dflash2").unwrap();
    let (_, d) = dflash2.resolve("gb10", None).unwrap();
    for (key, want) in [
        // 16, not the plain gate's 128: a DFlash2 serve REFUSES TO START
        // wider on GB10 — the verify pool is gamma-sized and its slot count
        // is pinned at 32 for any bs>=32, while the f16-pool relief that lets
        // MTP reach 128 is rejected with --dflash by design. The BENCH.toml
        // note carries the measured reserve table. Pinned here so the cap
        // cannot be quietly raised into a serve that will not boot.
        ("max_batch_size", "16"),
        ("kv_cache_dtype", "fp8"),
        ("ssm_cache_slots", "8"),
        ("max_model_len", "4096"),
        ("dflash", "true"),
        ("draft_model", "incoai/Qwen3.8-27B-DFlash2"),
        ("dflash_gamma", "8"),
    ] {
        assert_eq!(
            d.serve_overrides.get(key).map(String::as_str),
            Some(want),
            "concurrency-sweep-dflash2 serve pin {key}: {:?}",
            d.serve_overrides
        );
    }
    assert_eq!(d.serve_overrides.len(), 7, "{:?}", d.serve_overrides);
    assert!(
        !d.serve_overrides.contains_key("speculative"),
        "--dflash conflicts with --speculative at the CLI: pinning both would not start"
    );
    // The one-variable rule, asserted rather than described — with exactly
    // one documented exception. Every key the plain gate pins must be pinned
    // identically here EXCEPT max_batch_size, which the drafter's memory
    // footprint forces down (see the BENCH.toml note and its measured reserve
    // table). Listing the exception rather than skipping the check is the
    // point: a second axis of difference must never appear silently.
    const FORCED_BY_THE_DRAFTER: [&str; 1] = ["max_batch_size"];
    for (key, want) in &c.serve_overrides {
        if FORCED_BY_THE_DRAFTER.contains(&key.as_str()) {
            assert_ne!(
                d.serve_overrides.get(key),
                Some(want),
                "{key} is listed as forced apart by the drafter but the two gates agree on \
                 it — drop it from the exception list rather than leaving a stale excuse"
            );
            continue;
        }
        assert_eq!(
            d.serve_overrides.get(key),
            Some(want),
            "the two concurrency ladders must differ only in the drafter and the batch cap \
             it forces, but {key} differs"
        );
    }

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
