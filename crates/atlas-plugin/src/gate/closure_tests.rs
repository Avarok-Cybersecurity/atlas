// SPDX-License-Identifier: AGPL-3.0-only

//! Rung 0 is a mechanism for *not* re-running benchmarks, so most of these
//! tests are about the cases where it must refuse to.

use super::*;

fn tmp(name: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("atlas-gate-closure-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Two models on one hardware. `modelA` shadows `shared.cu` outright;
/// `modelB` inherits it. This is the shape the whole design turns on.
fn fixture(name: &str) -> std::path::PathBuf {
    let root = tmp(name);
    let hw = root.join("kernels/gb10");
    std::fs::create_dir_all(hw.join("common")).unwrap();
    std::fs::create_dir_all(hw.join("modelA/nvfp4")).unwrap();
    std::fs::create_dir_all(hw.join("modelB/nvfp4")).unwrap();
    std::fs::write(
        hw.join("HARDWARE.toml"),
        "[hardware]\nvendor = \"nvidia\"\n",
    )
    .unwrap();
    std::fs::write(hw.join("modelA/MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(hw.join("modelB/MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(
        hw.join("common/shared.cu"),
        "__global__ void s() { int t = 64; }\n",
    )
    .unwrap();
    std::fs::write(
        hw.join("modelA/nvfp4/shared.cu"),
        "__global__ void s() { /*A*/ }\n",
    )
    .unwrap();
    root
}

fn attest_all(root: &std::path::Path) -> Attestation {
    attest(root, "sm_121a", "nvcc 13.0.2", &BTreeMap::new())
}

const SHARED: &str = "kernels/gb10/common/shared.cu";

// ---------------------------------------------------------------------------
// The point of the mechanism
// ---------------------------------------------------------------------------

/// Nothing edited: every target still hashes the same, so the record stands.
#[test]
fn an_untouched_tree_excuses_a_kernel_path() {
    let root = fixture("untouched");
    let a = attest_all(&root);
    assert!(excuses(&root, &[SHARED.to_string()], &a));
}

/// ★ The saving. `modelA` genuinely shadows `shared.cu` without including it,
/// so a change to the common copy cannot reach it — and it keeps its record
/// while `modelB` loses one.
#[test]
fn a_shared_edit_re_opens_only_the_targets_that_compile_it() {
    let root = fixture("shared-edit");
    let a = attest_all(&root);
    std::fs::write(root.join(SHARED), "__global__ void s() { int t = 128; }\n").unwrap();

    let changed = changed_targets(&root, &[SHARED.to_string()], &a);
    assert_eq!(changed, vec!["gb10/modelB/nvfp4"], "only the inheritor");
    assert!(
        !excuses(&root, &[SHARED.to_string()], &a),
        "one affected target changed, so the gate re-opens"
    );
}

/// ★ REFUTATION 1 in the gate's own terms: a shadow that INCLUDES the common
/// file must lose its record too. Against a file-set design `modelA` would
/// keep it, and the edited bytes would ship ungated.
#[test]
fn a_shadow_that_includes_the_common_file_is_not_excused() {
    let root = fixture("include-shadow");
    std::fs::write(
        root.join("kernels/gb10/modelA/nvfp4/shared.cu"),
        "#include \"../../common/shared.cu\"\n",
    )
    .unwrap();
    let a = attest_all(&root);
    std::fs::write(root.join(SHARED), "__global__ void s() { int t = 128; }\n").unwrap();

    let mut changed = changed_targets(&root, &[SHARED.to_string()], &a);
    changed.sort();
    assert_eq!(
        changed,
        vec!["gb10/modelA/nvfp4", "gb10/modelB/nvfp4"],
        "the including shadow must re-open as well"
    );
}

/// ★ REFUTATION 2: headers are in no source set, so only the include walk can
/// see them.
#[test]
fn editing_an_included_header_is_not_excused() {
    let root = fixture("header");
    std::fs::write(root.join("kernels/gb10/common/tune.cuh"), "#define BR 64\n").unwrap();
    std::fs::write(
        root.join(SHARED),
        "#include \"tune.cuh\"\n__global__ void s() {}\n",
    )
    .unwrap();
    let a = attest_all(&root);

    std::fs::write(
        root.join("kernels/gb10/common/tune.cuh"),
        "#define BR 128\n",
    )
    .unwrap();
    let header = "kernels/gb10/common/tune.cuh".to_string();
    assert_eq!(
        changed_targets(&root, std::slice::from_ref(&header), &a),
        ["gb10/modelB/nvfp4"],
        "the inheriting target compiles the header through shared.cu"
    );
    assert!(
        !excuses(&root, &[header], &a),
        "a header edit must re-open the targets that include it"
    );
}

/// A tuned constant in `MODEL.toml` is compiled in, so it must invalidate the
/// model that reads it — and only that one.
#[test]
fn a_model_toml_edit_re_opens_only_that_model() {
    let root = fixture("model-toml");
    let a = attest_all(&root);
    std::fs::write(
        root.join("kernels/gb10/modelA/MODEL.toml"),
        "[behavior]\nthinking_default = true\n",
    )
    .unwrap();
    let path = "kernels/gb10/modelA/MODEL.toml".to_string();
    assert_eq!(
        changed_targets(&root, std::slice::from_ref(&path), &a),
        vec!["gb10/modelA/nvfp4"]
    );
    assert!(!excuses(&root, &[path], &a));
}

// ---------------------------------------------------------------------------
// Everything it must refuse to excuse
// ---------------------------------------------------------------------------

/// Records written before this existed carry nothing, and must behave exactly
/// as they did before — no silent upgrade to "covered".
#[test]
fn a_record_with_no_attestation_excuses_nothing() {
    let root = fixture("no-attestation");
    assert!(!excuses(&root, &[SHARED.to_string()], &Attestation::new()));
}

/// A target the record never mentioned cannot be vouched for. This is the
/// new-model case: it did not exist when the record was written.
#[test]
fn a_target_missing_from_the_attestation_is_not_excused() {
    let root = fixture("new-model");
    let mut a = attest_all(&root);
    a.remove("gb10/modelB/nvfp4");
    assert!(
        !excuses(&root, &[SHARED.to_string()], &a),
        "an unmentioned affected target must not be excused"
    );
}

/// ★ Only `kernels/` is in scope. A host-code path in the set ends the
/// question, even alongside kernel paths that would be excused alone.
#[test]
fn a_non_kernel_path_is_never_excused() {
    let root = fixture("host-path");
    let a = attest_all(&root);
    assert!(!excuses(
        &root,
        &["crates/spark-model/src/lib.rs".to_string()],
        &a
    ));
    assert!(
        !excuses(&root, &[SHARED.to_string(), "Cargo.lock".to_string()], &a),
        "one out-of-scope path must veto the whole set"
    );
}

/// A kernel path under no known target — a new hardware dir, a rename — is
/// unknown, and unknown is not excused.
#[test]
fn a_kernel_path_mapping_to_no_target_is_not_excused() {
    let root = fixture("unknown-target");
    let a = attest_all(&root);
    assert!(!excuses(
        &root,
        &["kernels/newhw/common/x.cu".to_string()],
        &a
    ));
}

/// Deleting a header a source includes is a real change and must re-open the
/// gate. It no longer makes the hash uncomputable — `atlas_closure` records
/// unresolvable includes rather than failing — but the recorded set is itself
/// hashed, so the digest moves either way.
#[test]
fn a_vanished_header_re_opens_the_gate() {
    let root = fixture("vanished-header");
    std::fs::write(root.join("kernels/gb10/common/tune.cuh"), "#define BR 64\n").unwrap();
    std::fs::write(
        root.join(SHARED),
        "#include \"tune.cuh\"\n__global__ void s() {}\n",
    )
    .unwrap();
    let a = attest_all(&root);

    std::fs::remove_file(root.join("kernels/gb10/common/tune.cuh")).unwrap();
    assert!(!excuses(&root, &[SHARED.to_string()], &a));
    assert_eq!(
        changed_targets(&root, &[SHARED.to_string()], &a),
        ["gb10/modelB/nvfp4"],
        "the target that included it is REPORTED, not silently dropped"
    );
}

/// A target whose sources cannot be resolved at all — the vendor table went
/// stale, the directory moved — is unknown, and unknown is never excused.
#[test]
fn a_target_whose_sources_do_not_resolve_is_not_excused() {
    let root = fixture("no-sources");
    let a = attest_all(&root);
    std::fs::write(
        root.join("kernels/gb10/HARDWARE.toml"),
        "[hardware]\nvendor = \"quantum-abacus\"\n",
    )
    .unwrap();
    assert!(!excuses(&root, &[SHARED.to_string()], &a));
}

/// An empty path list means the caller had nothing to excuse; answering "yes"
/// would be a vacuous truth that reads as a positive result at the call site.
#[test]
fn an_empty_path_list_excuses_nothing() {
    let root = fixture("empty-paths");
    assert!(!excuses(&root, &[], &attest_all(&root)));
}

/// The attestation records the inputs it was computed under and reuses them.
/// If it instead recomputed with the CI machine's toolchain, every record
/// would be invalidated by the checker's own environment.
#[test]
fn the_check_reuses_the_recorded_inputs_not_the_current_environment() {
    let root = fixture("inputs");
    // Attested under a toolchain no machine here has. Verification must still
    // succeed: it recomputes under the RECORDED inputs. Were it to substitute
    // the checker's own environment, every record would be invalidated by the
    // machine that happened to run CI.
    let exotic = attest(&root, "sm_999z", "nvcc from the future", &BTreeMap::new());
    assert!(excuses(&root, &[SHARED.to_string()], &exotic));

    // And the inputs really are inside the digest: the same sources attested
    // under a different arch produce a different hash, so a record cannot be
    // verified against a build it did not describe.
    let native = attest(&root, "sm_121a", "nvcc 13.0.2", &BTreeMap::new());
    assert_ne!(
        exotic["gb10/modelB/nvfp4"].hash, native["gb10/modelB/nvfp4"].hash,
        "arch and compiler must be inside the digest"
    );
}

/// Per-target nvcc flags come from `KERNEL.toml`, so they cannot be stored once
/// per record — a target that tunes its own flags would be checked against
/// another target's.
#[test]
fn per_target_flags_are_carried_per_target() {
    let root = fixture("flags");
    let flags = BTreeMap::from([("gb10/modelA/nvfp4".to_string(), vec!["-O3".to_string()])]);
    let a = attest(&root, "sm_121a", "nvcc 13.0.2", &flags);
    assert_eq!(a["gb10/modelA/nvfp4"].flags, vec!["-O3"]);
    assert!(a["gb10/modelB/nvfp4"].flags.is_empty());
    assert!(
        excuses(&root, &[SHARED.to_string()], &a),
        "recomputation must use each target's own flags"
    );
}
