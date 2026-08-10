// SPDX-License-Identifier: AGPL-3.0-only

//! The contract between the two halves of the closure hash.
//!
//! `atlas-kernels/build.rs` computes a target's sources with `collect_cu_files`
//! and bakes the hash into the binary. The gate recomputes them with
//! `atlas_plugin::gate::taxon::sources` and compares. Those are two separate
//! implementations of one rule, and if they ever disagree the hashes never
//! match, every record is invalidated forever, and the failure is INVISIBLE —
//! it looks exactly like "the kernels changed", which is the normal case.
//!
//! Fail-closed is the right direction, but a silently dead feature is still a
//! dead feature. This test is the only place the two halves are compared.

use std::path::{Path, PathBuf};

use atlas_plugin::gate::closure::{Attestation, TargetClosure};
use atlas_plugin::gate::taxon;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

/// ★ The load-bearing test: for a binary built with real kernels, every baked
/// hash must reproduce from the tree.
///
/// Skipped when the binary carries no attestation — an `ATLAS_SKIP_BUILD=1`
/// build compiled nothing, so there is nothing to agree with. That means CI's
/// no-GPU leg cannot run this; the GPU build is where it bites, which is also
/// where the disagreement would matter.
#[test]
fn the_baked_attestation_reproduces_from_the_tree() {
    let baked: Attestation = serde_json::from_str(atlas_kernels::TARGET_CLOSURES)
        .expect("TARGET_CLOSURES must parse into the type the gate reads");
    if baked.is_empty() {
        eprintln!(
            "no baked attestation (skip build) — the build-vs-gate agreement is \
             unverified in this run; it is checked on a real kernel build"
        );
        return;
    }

    let root = repo_root();

    // ★ Every target the build resolved must be attested, not merely every
    // attested target verifiable. Checking only the baked keys is how
    // `gb10/qwen3.6-27b/nvfp4` — the MLPerf flagship — went missing unnoticed:
    // 21 of 22 entries verified, the test passed, and the one target the whole
    // scheme exists to spare had no attestation at all. A silently absent entry
    // is fail-closed, which is why nothing else catches it.
    let gb10: Vec<_> = taxon::walk(&root)
        .into_iter()
        .filter(|t| t.hardware == "gb10" && t.quant == "nvfp4")
        .collect();
    let missing: Vec<String> = gb10
        .iter()
        .map(|t| t.to_string())
        .filter(|k| !baked.contains_key(k))
        .collect();
    assert!(
        missing.is_empty(),
        "the build resolved these targets but attested none of them, so their \
         records can never be excused: {missing:?}. Check build.rs's cargo:warning \
         output for the reason."
    );

    let mut checked = 0;
    for (key, recorded) in &baked {
        let parts: Vec<&str> = key.split('/').collect();
        assert_eq!(
            parts.len(),
            3,
            "attestation key must be hw/model/quant: {key}"
        );
        let target = taxon::Target {
            hardware: parts[0].into(),
            model: parts[1].into(),
            quant: parts[2].into(),
        };

        let sources = taxon::sources(&root, &target).unwrap_or_else(|| {
            panic!(
                "{key}: build.rs resolved sources for this target but taxon::sources \
                 did not. The two source resolvers have drifted, and the gate can \
                 never excuse this target again."
            )
        });
        let current = atlas_closure::hash(
            &root,
            &atlas_closure::ClosureInputs {
                sources,
                configs: taxon::configs(&root, &target),
                flags: recorded.flags.clone(),
                arch: recorded.arch.clone(),
                compiler: recorded.compiler.clone(),
            },
        )
        .unwrap_or_else(|e| panic!("{key}: recomputation failed: {e}"));

        assert_eq!(
            current, recorded.hash,
            "{key}: the tree-side hash disagrees with the one baked at build \
             time. Either the source sets differ (collect_cu_files vs \
             taxon::sources) or the config lists do — not a kernel change, a \
             bug in one of the two resolvers."
        );
        checked += 1;
    }
    assert!(checked > 0, "an attestation with no usable entries");
    // Printed rather than merely asserted: "3 passed" looks identical whether
    // this verified 22 targets or 1, and the count is the only thing that
    // distinguishes a real check from a nearly-empty one.
    eprintln!("closure attestation: {checked} target(s) reproduced from the tree");
}

/// The JSON build.rs writes by hand must deserialize into the struct the gate
/// reads. A renamed field would parse to nothing and disable the feature
/// silently, so the exact shape is pinned here.
#[test]
fn the_baked_json_shape_matches_what_the_gate_deserializes() {
    let sample = r#"{"gb10/qwen3.6-27b/nvfp4":{
        "hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "arch":"sm_121a",
        "compiler":"nvcc release 13.0, V13.0.2",
        "flags":["-lineinfo"]}}"#;
    let parsed: Attestation = serde_json::from_str(sample).expect("shape must match");
    let entry: &TargetClosure = &parsed["gb10/qwen3.6-27b/nvfp4"];
    assert_eq!(entry.arch, "sm_121a");
    assert_eq!(entry.flags, vec!["-lineinfo"]);
    assert!(entry.hash.starts_with("0123"));
}

/// A missing `flags` key is normal — most targets add none — but a missing
/// `hash` or `arch` must NOT quietly default, because a defaulted hash would
/// compare equal to another defaulted hash.
#[test]
fn a_malformed_entry_refuses_to_deserialize_rather_than_defaulting() {
    let no_hash = r#"{"gb10/m/q":{"arch":"sm_121a","compiler":"nvcc"}}"#;
    assert!(
        serde_json::from_str::<Attestation>(no_hash).is_err(),
        "an entry without a hash must not parse — a defaulted hash would match \
         every other defaulted hash"
    );
    let no_flags = r#"{"gb10/m/q":{"hash":"ab","arch":"sm_121a","compiler":"nvcc"}}"#;
    assert!(
        serde_json::from_str::<Attestation>(no_flags).is_ok(),
        "absent flags is the common case and must parse as empty"
    );
}
