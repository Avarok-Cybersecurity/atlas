// SPDX-License-Identifier: AGPL-3.0-only

// Permanent witness that `cfg(avarok_rdma_verbs)` survived the move of the
// verbs shim into the `avarok-rdma` crate.
//
// `rustc-cfg` does not propagate across crates: spark-storage's build.rs must
// re-emit the cfg from avarok-rdma's `links` metadata
// (DEP_AVAROK_RDMA_SHIM_HAS_VERBS). If that re-emit ever silently breaks, every
// `#[cfg(avarok_rdma_verbs)]` module in this crate compiles OUT while `cargo
// check` stays green — these tests turn that into a `cargo test -p
// spark-storage --lib` failure on verbs hosts.

/// On a verbs host (Linux + rdma-core, AVAROK_SKIP_BUILD unset) this test
/// EXISTS only when the cfg is on for this crate, and then asserts the
/// always-compiled witness agrees. On skip/macOS builds the cfg is off and
/// the test compiles out — that is the documented OFF state, not a failure.
#[cfg(avarok_rdma_verbs)]
#[test]
fn verbs_cfg_is_on_for_spark_storage() {
    assert!(crate::rdma_verbs_enabled());
    // And it must agree with the upstream crate that owns the shim: if
    // avarok-rdma built the shim but our re-emit vanished, the module gates
    // in THIS crate silently disagree with avarok-rdma's.
    assert!(avarok_rdma::verbs_enabled());
}

/// Both crates must always agree on the cfg, in BOTH states (they build in the
/// same cargo invocation with the same env + target, and the decision is made
/// in exactly one place: avarok-rdma/build.rs).
#[test]
fn verbs_cfg_agrees_with_avarok_rdma() {
    assert_eq!(crate::rdma_verbs_enabled(), avarok_rdma::verbs_enabled());
}
