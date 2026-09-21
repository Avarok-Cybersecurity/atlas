// SPDX-License-Identifier: AGPL-3.0-only

//! The Metal benchmark gate is OPEN — asserted, not merely absent.
//!
//! The owner asked for a benchmark set that becomes required when Apple Metal
//! kernels are edited, added gradually, and left open for now. "Open" has to be
//! a checked state rather than a gap, because a gap is indistinguishable from
//! nobody having thought about it, and because the obvious ways to quiet CI
//! later — a thresholds table, a `[benchmarks.limits]` block, a
//! promotion-candidate row — each convert an open gate into a number nobody
//! measured.
//!
//! A first attempt at this added `kernels/metal/<model>/BENCH.toml` with three
//! invented gate ids and fields (`id`, `status`, `what`) that are not the
//! schema. Eight tests in this crate failed: a BENCH.toml entry names a
//! REGISTERED gate plus a checkpoint and a recipe, so inventing gate ids
//! creates the "debt row nobody can ever discharge" that `coverage.rs:863`
//! describes. The open state needs no new file; it needs these assertions.

/// Metal declares no `[benchmarks.limits]`, and that is load-bearing.
///
/// `bench_certify` bails for a hardware class that declares none, naming the
/// missing thermal envelope and memory floor — so the certification path
/// REFUSES for Metal rather than passing vacuously. `hardware/limits.rs`
/// already pins the absence; this test records WHY it must stay absent, so a
/// future reader does not "fix" the omission and silently open the gate.
#[test]
fn metal_declares_no_benchmark_limits_and_that_is_deliberate() {
    // Computed here rather than reaching into `bench::bench_tests`, which is
    // private: widening a module's visibility so a test can borrow a helper is
    // a change to the crate's API for the convenience of a test.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/avarok-plugin -> repo root")
        .to_path_buf();
    let limits = crate::hardware::limits::limits(&root, "metal").expect("readable");
    assert!(
        limits.is_none(),
        "kernels/metal/HARDWARE.toml has acquired a [benchmarks.limits] block. \
         That is the switch that turns the Metal certification path from \
         REFUSING to RUNNING, and it must not be flipped before the thermal \
         envelope and memory floor are measured on the 48 GiB box. If you \
         measured them, say so in the commit and delete this test's premise \
         deliberately rather than as a side effect."
    );
}

/// No Metal benchmark id is registered as required, not-required, or a
/// promotion candidate.
///
/// `coverage_promotion_tests.rs` already refuses a candidate naming an
/// unregistered benchmark. This asserts the other direction: nothing Metal has
/// been slipped into the registries, which is what would make the gate look
/// live while measuring nothing.
#[test]
fn no_metal_benchmark_is_registered_anywhere_yet() {
    // REQUIRED is [GateCoverage; 12] and PROMOTION_CANDIDATES is
    // &[GateCoverage]; both carry the benchmark id in `.id`.
    let named: Vec<&str> = crate::gate::coverage::REQUIRED
        .iter()
        .chain(crate::gate::coverage::PROMOTION_CANDIDATES.iter())
        .map(|g| g.id)
        .filter(|id| id.contains("metal"))
        .collect();
    assert!(
        named.is_empty(),
        "a Metal benchmark id appeared in REQUIRED or PROMOTION_CANDIDATES: \
         {named:?}. The closing sequence is measure the envelope -> declare \
         [benchmarks.limits] -> register a driver -> list the candidate -> make \
         it required, in that order. Listing the id first is a debt row nobody \
         can discharge."
    );
}
