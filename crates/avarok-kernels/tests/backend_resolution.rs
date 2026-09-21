// SPDX-License-Identifier: AGPL-3.0-only

//! The backend-selection table, pinned.
//!
//! `build_backend::resolve` decides whether a build gets `avarok_cuda`,
//! `avarok_metal`, both or neither. It runs inside five build scripts, where
//! nothing can reach it, so the rule lives in its own file and this test is
//! the only thing that reads it twice.

// `emit` and `register_cfgs` are for the build scripts; this test exercises
// only the pure rule, and the workspace denies dead_code.
#[allow(dead_code)]
#[path = "../build_backend.rs"]
mod build_backend;

use build_backend::resolve;

#[test]
fn the_four_feature_times_target_combinations() {
    // (cuda feature, target_os) -> (avarok_cuda, avarok_metal)
    let cases = [
        // Linux, features as shipped: a CUDA build, exactly as today.
        ((true, "linux"), (true, false)),
        // macOS with the cuda feature still on -- which is the DEFAULT, since
        // `default = ["cuda"]` and a feature cannot be made target-conditional.
        // This is the case the whole design exists for: the feature resolves,
        // `cudarc` is excluded by a [target.'cfg(not(target_os = "macos"))']
        // table, and avarok_cuda must be OFF so the 130 cfg(feature = "cuda")
        // blocks do not activate against a dependency that is not there.
        ((true, "macos"), (false, true)),
        // --no-default-features on Linux: neither backend.
        ((false, "linux"), (false, false)),
        // --no-default-features on macOS: Metal anyway. The requirement is that
        // Metal needs no flag, which includes needing no feature.
        ((false, "macos"), (false, true)),
    ];
    for ((feat, os), want) in cases {
        assert_eq!(
            resolve(feat, os),
            want,
            "resolve(feature_cuda = {feat}, target_os = {os:?})"
        );
    }
}

/// The control. If `resolve` ever ignores the target and keys only on the
/// feature, the table above still passes for three of its four rows — so the
/// assertion that matters is the one that separates the two macOS rows from
/// their Linux twins.
#[test]
fn the_target_os_is_load_bearing_not_decorative() {
    assert_ne!(
        resolve(true, "linux"),
        resolve(true, "macos"),
        "the cuda feature must resolve DIFFERENTLY on macOS; if these are equal, \
         the rule is reading the feature and ignoring the target, which is the \
         bug that makes a flagless macOS build fail on cudarc"
    );
    assert!(
        !resolve(true, "macos").0,
        "avarok_cuda must be OFF on macOS even with the cuda feature on"
    );
    assert!(
        resolve(false, "macos").1,
        "avarok_metal must be ON on macOS even with no features at all"
    );
}
