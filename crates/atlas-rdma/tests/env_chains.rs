// SPDX-License-Identifier: AGPL-3.0-only

// Pins the TWO distinct env-fallback semantics the RDMA tiers deploy with:
// `first_set` (an exported-but-EMPTY var counts as set — KV/expert/snapshot
// reads and the LoRA DEV chain) vs
// `first_nonempty` (empty is SKIPPED — the weight tier), plus the
// parse-or-fall-through `first_set_u32`. A unified helper that flipped either
// behavior would silently change deployed configs (e.g. make
// ATLAS_WEIGHT_RDMA_GID start affecting LoRA).

use atlas_rdma::env::{first_nonempty, first_set, first_set_u32};

/// All env mutation lives in this ONE test: `set_var` is process-global and
/// test threads run concurrently, so a single serialized test avoids races.
/// Var names are unique to this file.
#[test]
fn env_helper_semantics() {
    // SAFETY: single-threaded within this test; the vars are test-unique, and
    // no other test in this binary mutates the environment.
    unsafe {
        std::env::set_var("ATLAS_RDMATEST_EMPTY", "");
        std::env::set_var("ATLAS_RDMATEST_DEV2", "rocep1s0f1");
        std::env::set_var("ATLAS_RDMATEST_BADNUM", "not-a-number");
        std::env::set_var("ATLAS_RDMATEST_NUM", "7");
    }

    // first_set: an exported-but-empty var COUNTS AS SET (Result::or_else /
    // unwrap_or_else chain semantics) — it wins over later keys AND defaults.
    assert_eq!(
        first_set(&["ATLAS_RDMATEST_EMPTY", "ATLAS_RDMATEST_DEV2"], "dflt"),
        "",
        "first_set must treat an exported-but-empty var as set"
    );
    assert_eq!(
        first_set(&["ATLAS_RDMATEST_UNSET", "ATLAS_RDMATEST_DEV2"], "dflt"),
        "rocep1s0f1",
        "first_set must chain past unset keys"
    );

    // first_nonempty: the same empty var is SKIPPED (weight-tier env_str).
    assert_eq!(
        first_nonempty(&["ATLAS_RDMATEST_EMPTY", "ATLAS_RDMATEST_DEV2"], "dflt"),
        "rocep1s0f1",
        "first_nonempty must skip an exported-but-empty var"
    );

    // first_set_u32: a set-but-unparseable var falls through to the next key
    // (and to the default when it is the only key) — every pre-extraction
    // per-client env_u32 behaved this way.
    assert_eq!(
        first_set_u32(&["ATLAS_RDMATEST_BADNUM", "ATLAS_RDMATEST_NUM"], 3),
        7
    );
    assert_eq!(first_set_u32(&["ATLAS_RDMATEST_BADNUM"], 3), 3);
    assert_eq!(first_set_u32(&["ATLAS_RDMATEST_NUM"], 3), 7);
}

/// Read-only (no env mutation): unset chains land on the tier defaults —
/// the deployed single-rail path with a clean environment.
#[test]
fn unset_chains_yield_defaults() {
    assert_eq!(
        first_set(
            &["ATLAS_RDMATEST_NOPE1", "ATLAS_RDMATEST_NOPE2"],
            "roceP2p1s0f1"
        ),
        "roceP2p1s0f1"
    );
    assert_eq!(
        first_nonempty(&["ATLAS_RDMATEST_NOPE1"], "roceP2p1s0f1"),
        "roceP2p1s0f1"
    );
    assert_eq!(first_set_u32(&["ATLAS_RDMATEST_NOPE1"], 3), 3);
}
