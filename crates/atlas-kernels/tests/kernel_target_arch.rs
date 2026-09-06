// SPDX-License-Identifier: AGPL-3.0-only

//! `kernel_target_arch` — the HARDWARE.toml `arch` → `KernelTarget.arch`
//! mapping, pinned.
//!
//! ORACLE: the CUDA toolkit's own naming. `nvcc -arch=` accepts a *feature*
//! architecture (`sm_90a`, `sm_100a`, `sm_121f`) whose trailing letter selects
//! the arch-specific / family-specific instruction set on top of a base SM
//! number. The base number is what `KernelTarget.arch` records — the constants
//! in `crates/atlas-core/src/target.rs` spell it `sm_121`, not `sm_121f` — so
//! the mapping must strip the suffix and nothing else. Non-NVIDIA arch strings
//! (SCALE `gfx*`, Metal `metal3.1`) are opaque to this rule and pass through.
//!
//! `build_codegen.rs` used `trim_end_matches('f')`, which handled `sm_121f`
//! only; `sm_90a` (Hopper) would have reached the registry as `sm_90a`.
//!
//! This is an integration test rather than a `#[cfg(test)]` module inside the
//! build script because cargo never runs a build script's own unit tests —
//! the same reason `tests/kernel_shadow_detector.rs` exists.

#[path = "../build_arch.rs"]
mod build_arch;

use build_arch::kernel_target_arch;

#[test]
fn family_specific_suffix_is_stripped() {
    // GB10 ships `arch = "sm_121f"`; the registry records `sm_121`.
    assert_eq!(kernel_target_arch("sm_121f"), "sm_121");
}

#[test]
fn arch_specific_suffix_is_stripped() {
    // Hopper ships `arch = "sm_90a"` (wgmma/TMA opt-in), Blackwell datacenter
    // `sm_100a`. Both record their base SM.
    assert_eq!(kernel_target_arch("sm_90a"), "sm_90");
    assert_eq!(kernel_target_arch("sm_100a"), "sm_100");
    assert_eq!(kernel_target_arch("sm_121a"), "sm_121");
}

#[test]
fn a_plain_sm_number_is_unchanged() {
    assert_eq!(kernel_target_arch("sm_121"), "sm_121");
    assert_eq!(kernel_target_arch("sm_90"), "sm_90");
}

#[test]
fn non_nvidia_arch_strings_pass_through() {
    // SCALE/HIP select a per-arch toolchain dir by this exact string, and
    // Metal forwards it to `-std=`. Neither may be rewritten.
    assert_eq!(kernel_target_arch("gfx1151"), "gfx1151");
    assert_eq!(kernel_target_arch("gfx90a"), "gfx90a");
    assert_eq!(kernel_target_arch("metal3.1"), "metal3.1");
}

// ── the pair a compiled target records ──

use build_arch::target_arch_fields;

/// ORACLE: `kernels/hopper/HARDWARE.toml` declares `arch = "sm_90a"`, and the
/// two fields a `TargetPtxSet` carries are the two readings of that one
/// declaration: `KernelTarget.arch` is the base SM the registry, the gate
/// baselines and `KernelTarget`'s own constants are keyed by, and `ptx_arch`
/// is the string nvcc was actually handed.
///
/// They are produced together so they cannot drift. The drift is not
/// hypothetical: the GPU preflight was reading `KernelTarget.arch`, where
/// `sm_90a` arrives as `sm_90` — plain PTX, forward-compatible by rule — so
/// `sm_90a` kernels PASSED the preflight on a CC 10.0 device and died inside
/// `cuModuleLoadData`, which is the failure the preflight exists to replace.
#[test]
fn a_target_records_both_the_base_sm_and_the_arch_nvcc_was_handed() {
    assert_eq!(
        target_arch_fields("sm_90a"),
        ("sm_90".to_string(), "sm_90a")
    );
    assert_eq!(
        target_arch_fields("sm_121f"),
        ("sm_121".to_string(), "sm_121f")
    );
    assert_eq!(
        target_arch_fields("sm_100a"),
        ("sm_100".to_string(), "sm_100a")
    );
}

/// A plain arch and a non-NVIDIA one record the SAME string twice — there is
/// no suffix to strip, so the base and the verbatim reading coincide. Pinned
/// because a caller reading `ptx_arch` must not have to ask whether a Metal or
/// SCALE build fills it differently.
#[test]
fn an_arch_with_no_feature_suffix_records_the_same_string_twice() {
    assert_eq!(target_arch_fields("sm_90"), ("sm_90".to_string(), "sm_90"));
    assert_eq!(
        target_arch_fields("gfx1151"),
        ("gfx1151".to_string(), "gfx1151")
    );
    assert_eq!(
        target_arch_fields("metal3.1"),
        ("metal3.1".to_string(), "metal3.1")
    );
}
