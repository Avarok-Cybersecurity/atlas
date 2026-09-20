// SPDX-License-Identifier: AGPL-3.0-only

//! The MoE ladder's descriptor: a third gate id over the one concurrency
//! driver, pinned to the same floor params as its two dense siblings and to
//! the MoE family it is defined on. Its BENCH.toml entry is pinned by value in
//! `gate::bench_override_tree_tests`; its promotion debt in
//! `gate::coverage_promotion_tests`.
//!
//! Its own file because `concurrency_tests.rs` and
//! `concurrency_verdict_tests.rs` both sit at the 500-line cap; registered in
//! `gate::coverage::TEST_ONLY_RUST_MODULES` so an edit here re-opens no record.

use super::*;

/// Three gate ids, one driver: a record of any of them is scored on the same
/// metric names, or a floor written on `c8_aggregate_tok_s` for the MoE would
/// silently gate nothing.
#[test]
fn the_moe_gate_shares_the_drivers_floor_pairing() {
    assert_eq!(MOE_DESCRIPTOR.id, "concurrency-sweep-moe");
    assert_eq!(
        MOE_DESCRIPTOR.threshold_params, DESCRIPTOR.threshold_params,
        "the MoE ladder must gate on the same metric names as the dense one"
    );
    assert_eq!(
        crate::registry::find("concurrency-sweep-moe").map(|d| d.id),
        Some("concurrency-sweep-moe"),
        "an unregistered gate id can be neither run nor owed"
    );
    // The ctor is the shared driver: a `prompt_mode = "essay"` pin lands on
    // the same `Fixture::Essay` the published-instrument digests are pinned
    // against, so the MoE cell's request is the harness's byte for byte.
    let mut b = ConcurrencySweep::default();
    let mut v = ParamValues::defaults(&b.parameters());
    v.set("prompt_mode", ParamValue::Text("essay".into()));
    v.set("concurrencies", ParamValue::IntList(vec![1, 2, 4, 8, 16]));
    v.set("isls", ParamValue::IntList(vec![128]));
    v.set("osl", ParamValue::Int(1024));
    b.configure(&v).unwrap();
    assert_eq!(b.fixture, Fixture::Essay);
    assert_eq!(b.osl, 1024);
    assert_eq!(
        b.cells,
        vec![(128, 1), (128, 2), (128, 4), (128, 8), (128, 16)]
    );
    assert!(!b.floors.gating(), "an unmeasured entry fills no floor");
}

/// Defined on the MoE family, and the family test is a substring one: the
/// FP8 flagship (this gate's subject) and the NVFP4 sibling are both in, the
/// dense 27B is out.
#[test]
fn the_moe_gate_is_defined_on_the_moe_family_only() {
    let expect = MOE_DESCRIPTOR
        .intended_for
        .expect("the MoE ladder names the family it is defined on");
    assert!(expect.accepts("Qwen/Qwen3.6-35B-A3B-FP8"));
    assert!(expect.accepts("nvidia/Qwen3.6-35B-A3B-NVFP4"));
    assert!(!expect.accepts("unsloth/Qwen3.8-27B-NVFP4"));
    assert!(!expect.accepts("unsloth/Qwen3.6-27B-NVFP4"));
    assert!(
        DESCRIPTOR.intended_for.is_none(),
        "the plain gate stays family-agnostic; only the MoE and DFlash2 ids are pinned"
    );
}
