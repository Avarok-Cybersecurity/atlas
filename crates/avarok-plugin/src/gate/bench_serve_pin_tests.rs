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
/// else. The echolp pin does not move the floors: those are the ratchet of
/// the gate's measurement definition, and a pin is capacity (Marconi pool),
/// not a score lever. The floors themselves are pinned here too, so a move
/// is a recorded decision, never a drive-by: 86.50/86.90 → 84.56/85.77 on
/// 2026-09-14 (issue #1083 — one tool prompt, not two).
#[test]
fn the_trees_serve_pins_sit_on_the_gates_that_need_them() {
    let root = repo_root();

    // Gate B: the 35B echolp draw self-starts with the Marconi pool pinned, so
    // a 1004-sample serial generate cannot evict its own snapshots — with the
    // floors exactly where BENCH.toml's note records them.
    let echolp = baseline_for(&root, "bfcl-subset-echolp").unwrap();
    let (_, e) = echolp.resolve("gb10", None).unwrap();
    assert_eq!(e.metrics["overall_accuracy"].min, Some(84.56));
    assert_eq!(e.metrics["normalized_single_turn_score"].min, Some(85.77));
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

    // The concurrency gate no longer declares a serve profile at all: it names
    // the recipe that IS the profile. Until 2026-09-22 it served the shared
    // AGENTIC recipe — a serial reproduction config (batch 1, bf16 KV, 256
    // Marconi slots, 32K context) that strangles a concurrency instrument —
    // and overrode it seventeen keys at a time, which still left two of that
    // recipe's defaults inherited in silence (`lm_head_dtype: bf16` and
    // `kv_high_precision_layers: auto`). Marconi is no longer pinned either:
    // the 32-slot rule was calibrated on the RETIRED isl-512 instrument, and at
    // this gate's isl 128 (~200 rendered tokens) every snapshot restore is
    // declined under DEFAULT_MARCONI_MIN_TOKENS = 256, so the recipe's 8 —
    // which is also the published leg's value — costs no warm behaviour and
    // returns ~3.55 GiB to the KV budget.
    let sweep = baseline_for(&root, "concurrency-sweep").unwrap();
    let (_, c) = sweep.resolve("gb10", None).unwrap();
    // ★ SEVENTEEN PINS -> THREE, 2026-09-22 (owner: "the gate serves the
    // throughput recipe ... match the throughput recipe"). The entry now names
    // the THROUGHPUT recipe, which is bench/ladder38/published.json
    // `series[0].cli` frozen as a file -- verified key for key. Fourteen pins
    // were therefore re-stating that recipe's own defaults back at it and are
    // gone; what they used to say is in the BENCH.toml block, with the A/B that
    // justified the re-point (+41/45/53% at C=8/32/128).
    //
    // This assertion is the guard on the re-point itself: the gate had been
    // serving the AGENTIC recipe and inheriting its `lm_head_dtype: bf16` in
    // silence, which above M=8 (lm_head_batchm_max) drops the batched GEMV for
    // a scalar dense_gemm over a 248K vocab -- a per-sequence cost, and the
    // shape of the measured deficit.
    assert_eq!(
        c.recipe.as_deref(),
        Some("qwen3.8/qwen3.8-27b-nvfp4-throughput"),
        "the concurrency ladder must serve the published leg's own recipe, not the \
         agentic profile it used to override key by key"
    );
    for (key, want) in [
        // These three restate recipe defaults ON PURPOSE. ladder-baselines.js
        // builds a record's fingerprint from `params` + `serve_overrides` and
        // nothing else -- it never reads `serve_resolved` -- and all three are
        // REQUIRED_AXES. Absent here they would read null, null counts as a
        // difference on a required axis, and the live series would stop being
        // comparable to the published bar: the exact defect the 2026-09-21
        // re-point existed to fix. So they are fingerprint pins, not serve pins,
        // and that is why they alone survived the cut.
        ("max_batch_size", "128"),
        ("kv_cache_dtype", "fp8"),
        ("max_model_len", "2048"),
    ] {
        assert_eq!(
            c.serve_overrides.get(key).map(String::as_str),
            Some(want),
            "concurrency-sweep serve pin {key}: {:?}",
            c.serve_overrides
        );
    }
    assert_eq!(c.serve_overrides.len(), 3, "{:?}", c.serve_overrides);
    assert!(
        !c.serve_overrides.contains_key("lm_head_dtype"),
        "the throughput recipe leaves the head at the checkpoint's native NVFP4; pinning \
         bf16 here is what cost 41-53% across the ladder"
    );
    assert!(
        !c.serve_overrides.contains_key("ssm_cache_slots"),
        "the recipe's 8 is also the published leg's `--ssm-cache-slots 8`; overriding it \
         back to 32 re-opens the last disagreement with that leg"
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
        ("ssm_cache_slots", "32"),
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
    // The one-variable rule, asserted rather than described — with exactly TWO
    // documented exceptions, each in its own list so the REASON a key differs
    // is recorded and not just the fact. Every other key the plain gate pins
    // must be pinned identically here. Listing an exception rather than
    // skipping the check is the point: a third axis of difference must never
    // appear silently, and a listed key that has quietly come back into
    // agreement fails too, so an excuse cannot outlive its cause.
    //
    // max_batch_size: forced down by the drafter's memory footprint (see the
    // BENCH.toml note and its measured reserve table).
    const FORCED_BY_THE_DRAFTER: [&str; 1] = ["max_batch_size"];
    // max_model_len: forced apart on 2026-09-21 by the PLAIN gate's instrument
    // re-point, not by anything about DFlash2. The plain ladder is now pinned
    // to the published Atlas-vs-vLLM instrument (ISL 128 / OSL 1024 / essay,
    // ctx 2048) so its live record can be drawn against the measured vLLM bar;
    // `max_model_len` is a REQUIRED axis of that fingerprint
    // (site/src/lib/ladder-baselines.js), so 2048 is not a free choice there.
    // This gate deliberately did NOT follow: DFlash2 is not on the published
    // ladder, its bars were cut at ctx 4096 / ISL 512 / OSL 200, and moving its
    // context would refuse every record it has (`check_record` demands the pin)
    // to buy a comparison nothing draws. The two ladders' shared rungs stopped
    // being directly comparable at the same moment, which is stated in
    // bench_override_tree_tests and in both BENCH.toml entries.
    const FORCED_BY_THE_REPOINT: [&str; 1] = ["max_model_len"];
    // ★ WHAT THIS RULE NO LONGER COVERS, stated because a narrowed test that
    // does not say it narrowed is worse than no test. Until 2026-09-22 the two
    // ladders shared the agentic recipe and differed only in their pins, so
    // iterating the plain gate's seventeen pins really did compare the two
    // serves. They now resolve DIFFERENT RECIPES -- throughput here,
    // qwen3.8-27b-nvfp4-dflash2 there -- and the plain gate pins three keys,
    // so this loop compares three. The MTP-policy and published-profile
    // exception lists that used to stand here are deleted rather than kept as
    // commentary: their keys are no longer pinned by the plain gate at all, so
    // as consts they would be dead code, and as excuses they would outlive
    // their cause -- which is the one thing this rule exists to forbid. The
    // divergence they described is now a property of the two recipes and is
    // asserted at its own source, by the `c.recipe` assertion above.
    for (key, want) in &c.serve_overrides {
        if FORCED_BY_THE_DRAFTER.contains(&key.as_str())
            || FORCED_BY_THE_REPOINT.contains(&key.as_str())
        {
            assert_ne!(
                d.serve_overrides.get(key),
                Some(want),
                "{key} is listed as forced apart but the two gates agree on it — drop it \
                 from the exception list rather than leaving a stale excuse"
            );
            continue;
        }
        assert_eq!(
            d.serve_overrides.get(key),
            Some(want),
            "the two concurrency ladders may differ only where an exception list says so \
             and says why, but {key} differs"
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

/// A hermetic gate must pin EVERYTHING hermetic closes, and it is refused at
/// parse if it does not.
///
/// `check_record` compares the record's serve overrides against the baseline's
/// pins in BOTH directions, and a requested `hermetic=true` expands into the
/// keys it closes — so the record carries three keys where this entry pins
/// one, and the check fails with "present on the record but not pinned by the
/// baseline". That failure lands AFTER the run, having spent the GPU hours on
/// a gate that could never have been discharged by the run it asked for. So it
/// is a parse error, in milliseconds, like the `port` refusal above.
#[test]
fn a_hermetic_gate_that_does_not_pin_what_hermetic_closes_is_refused() {
    let root = fixture(
        "hermetic-underpinned",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.serve_overrides]
hermetic = "true"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    let err = format!("{:#}", baseline_for(&root, "bfcl-subset").unwrap_err());
    assert!(
        err.contains("pins hermetic=true but not"),
        "must name the omission: {err}"
    );
    assert!(
        err.contains("enable_prefix_caching=false") && err.contains("mtp_gate=force"),
        "and must name every missing key at the value it needs: {err}"
    );
}

/// The complete set parses. A refusal that also rejected the CORRECT entry
/// would make the regime unusable in a baseline, which is the same class of
/// mistake as the validator that made `--hermetic` unusable in a recipe.
#[test]
fn a_hermetic_gate_that_pins_the_whole_set_is_accepted() {
    let root = fixture(
        "hermetic-complete",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.serve_overrides]
hermetic = "true"
enable_prefix_caching = "false"
mtp_gate = "force"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    let baseline = baseline_for(&root, "bfcl-subset").expect("the complete set must parse");
    let (_, entry) = baseline.resolve("gb10", None).unwrap();
    assert_eq!(entry.serve_overrides.len(), 3);
    assert!(crate::gate::hermetic::missing_pins(&entry.serve_overrides).is_empty());
}

/// And an entry that pins none of it is untouched — the rule must not fire on
/// every gate in the repository.
#[test]
fn a_gate_with_no_hermetic_pin_is_unaffected() {
    let root = fixture(
        "hermetic-absent",
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
    assert!(baseline_for(&root, "bfcl-subset").is_ok());
}
