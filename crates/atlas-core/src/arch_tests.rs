// SPDX-License-Identifier: AGPL-3.0-only

//! Compatibility matrix for [`super::ptx_arch_runs_on_device`].
//!
//! ORACLE for every case below: NVIDIA CUDA C++ Programming Guide,
//! "Application Compatibility" → *PTX Compatibility* and *Feature-specific /
//! Family-specific architecture features* (CUDA 12.9+). Three rules, and
//! nothing else:
//!
//! 1. Plain `sm_XY` PTX JIT-compiles on any device with CC >= X.Y.
//! 2. `sm_XYa` (architecture-specific) runs ONLY on CC == X.Y.
//! 3. `sm_XYf` (family-specific) runs on devices of the SAME MAJOR family
//!    with CC >= X.Y.
//!
//! Each test names which rule it pins.

use super::{ArchMismatch, SmSuffix, parse_sm_arch, ptx_arch_runs_on_device, target_hint};

/// Rule 3 — the shipped GB10 arch on the GB10 it was built for.
///
/// Oracle: `kernels/gb10/HARDWARE.toml` declares `arch = "sm_121f"` and
/// `compute_capability = "12.1"`, and family PTX admits CC >= its own.
#[test]
fn family_ptx_runs_on_the_device_it_was_built_for() {
    assert!(ptx_arch_runs_on_device("sm_121f", (12, 1)).is_ok());
}

/// Rule 3 — the customer bring-up this whole preflight exists for.
///
/// Oracle: an H100 is CC 9.0; `sm_121f` is family-specific to 12.x, so the
/// driver has no code for it and the load fails deep inside `cuModuleLoadData`.
#[test]
fn family_ptx_does_not_run_on_a_hopper_device() {
    let err = ptx_arch_runs_on_device("sm_121f", (9, 0)).expect_err("sm_121f cannot run on CC 9.0");
    assert_eq!(err.compiled_arch, "sm_121f");
    assert_eq!(err.device_cc, (9, 0));
}

/// Rule 3 — family membership is necessary but NOT sufficient.
///
/// Oracle: family PTX still requires CC >= the compiled CC, so `sm_121f` does
/// not run on a 12.0 part even though both are 12.x.
#[test]
fn family_ptx_does_not_run_below_its_own_compute_capability() {
    assert!(ptx_arch_runs_on_device("sm_121f", (12, 0)).is_err());
}

/// Rule 3 — a different major family is out, however new.
///
/// Oracle: `sm_121f` is scoped to the 12.x family; CC 10.0 (Blackwell
/// datacentre) is a different family, not merely an older one.
#[test]
fn family_ptx_does_not_run_on_a_different_family() {
    assert!(ptx_arch_runs_on_device("sm_121f", (10, 0)).is_err());
}

/// Rule 2 — the arch the Hopper target compiles, on Hopper.
#[test]
fn arch_specific_ptx_runs_on_its_exact_compute_capability() {
    assert!(ptx_arch_runs_on_device("sm_90a", (9, 0)).is_ok());
}

/// Rule 2 — `a` PTX is never forward-compatible, not even one step.
///
/// Oracle: architecture-specific features (wgmma, TMA descriptors) are not
/// re-emitted for later architectures; the guide states `sm_XYa` binaries run
/// only on CC X.Y.
#[test]
fn arch_specific_ptx_does_not_run_on_a_newer_architecture() {
    assert!(ptx_arch_runs_on_device("sm_90a", (10, 0)).is_err());
    assert!(ptx_arch_runs_on_device("sm_90a", (12, 1)).is_err());
}

/// Rule 1 — plain PTX is forward-compatible by JIT.
#[test]
fn plain_ptx_runs_on_its_own_and_on_newer_devices() {
    assert!(ptx_arch_runs_on_device("sm_90", (9, 0)).is_ok());
    assert!(ptx_arch_runs_on_device("sm_90", (12, 1)).is_ok());
    assert!(ptx_arch_runs_on_device("sm_80", (9, 0)).is_ok());
}

/// Rule 1 — forward, never backward.
#[test]
fn plain_ptx_does_not_run_on_an_older_device() {
    assert!(ptx_arch_runs_on_device("sm_121", (9, 0)).is_err());
}

/// A non-NVIDIA target string is not a CUDA compute capability question.
///
/// DECISION, documented on [`super::ptx_arch_runs_on_device`]: `gfx1151`
/// (SCALE/HIP) and `metal3.1` do not parse as `sm_*`, the `(major, minor)`
/// pair means nothing for them, and the check passes rather than inventing a
/// verdict. The caller learns "not an NVIDIA arch" from `parse_sm_arch`
/// returning `None`, which is the marker.
#[test]
fn a_non_nvidia_arch_is_not_judged_by_compute_capability() {
    for arch in ["gfx1151", "metal3.1", "", "sm_", "sm_9", "sm_12x"] {
        assert!(parse_sm_arch(arch).is_none(), "{arch} must not parse");
        assert!(
            ptx_arch_runs_on_device(arch, (9, 0)).is_ok(),
            "{arch} must not fail a CUDA compute-capability check"
        );
    }
}

/// The parser splits the LAST digit as the minor version — NVIDIA's own
/// convention (`sm_90` = 9.0, `sm_100` = 10.0, `sm_121` = 12.1).
#[test]
fn the_parser_splits_the_last_digit_as_the_minor_version() {
    let cases = [
        ("sm_80", 8, 0, SmSuffix::None),
        ("sm_90", 9, 0, SmSuffix::None),
        ("sm_90a", 9, 0, SmSuffix::Arch),
        ("sm_100", 10, 0, SmSuffix::None),
        ("sm_121f", 12, 1, SmSuffix::Family),
    ];
    for (text, major, minor, suffix) in cases {
        let got = parse_sm_arch(text).unwrap_or_else(|| panic!("{text} must parse"));
        assert_eq!((got.major, got.minor, got.suffix), (major, minor, suffix));
    }
}

/// The operator-facing message must name BOTH sides and the way out.
///
/// This is the whole point of the type: the driver's own error
/// (`CUDA_ERROR_NO_BINARY_FOR_GPU`) names neither the arch we compiled nor the
/// GPU in the box, so an operator hitting it has nothing to act on.
#[test]
fn the_mismatch_message_names_both_sides_and_the_fix() {
    let err: ArchMismatch =
        ptx_arch_runs_on_device("sm_121f", (9, 0)).expect_err("mismatch expected");
    let msg = err.to_string();
    assert!(msg.contains("sm_121f"), "names the compiled arch: {msg}");
    assert!(msg.contains("9.0"), "names the device CC: {msg}");
    assert!(
        msg.contains("ATLAS_TARGET_HW=hopper"),
        "hints the target that fits CC 9.0: {msg}"
    );
    assert!(
        msg.contains("HARDWARE.toml"),
        "points at the file to change: {msg}"
    );
    assert!(
        msg.contains("use the image built for this GPU"),
        "offers the no-rebuild way out: {msg}"
    );
}

/// The reverse bring-up: gb10 kernels are what a CC 12.1 box wants.
#[test]
fn the_mismatch_message_hints_gb10_for_a_twelve_one_device() {
    let msg = ptx_arch_runs_on_device("sm_90a", (12, 1))
        .expect_err("mismatch expected")
        .to_string();
    assert!(msg.contains("ATLAS_TARGET_HW=gb10"), "{msg}");
}

/// The third shipped target. Oracle: `kernels/b200/HARDWARE.toml` —
/// `compute_capability = "10.0"`, `arch = "sm_100a"`.
///
/// This is the hint an operator most needs and was least likely to get: CC
/// 10.0 is where BOTH other NVIDIA targets fail (`sm_121f` is a different
/// major family, `sm_90a` never travels forward), so before `kernels/b200`
/// existed a B200 got "no shipped target matches compute capability 10.0" from
/// every image Atlas published.
#[test]
fn a_blackwell_datacentre_device_is_pointed_at_the_b200_target() {
    assert_eq!(target_hint((10, 0)), Some("b200"));
    for compiled in ["sm_121f", "sm_90a"] {
        let msg = ptx_arch_runs_on_device(compiled, (10, 0))
            .expect_err("neither shipped arch runs on CC 10.0")
            .to_string();
        assert!(msg.contains("ATLAS_TARGET_HW=b200"), "{msg}");
    }
}

/// B300 / GB300 are SM 10.3 and Atlas compiles nothing for them. The hint must
/// stay silent rather than nominate `b200`: `sm_100a` is architecture-specific
/// and does not run on 10.3, so pointing an operator at that build would send
/// them to rebuild an image that fails the same way.
#[test]
fn blackwell_ultra_has_no_shipped_target_and_is_not_pointed_at_b200() {
    assert_eq!(target_hint((10, 3)), None);
    let msg = ptx_arch_runs_on_device("sm_100a", (10, 3))
        .expect_err("sm_100a cannot run on CC 10.3")
        .to_string();
    assert!(msg.contains("no shipped target"), "{msg}");
    assert!(!msg.contains("ATLAS_TARGET_HW="), "{msg}");
}

/// A CC with no shipped target says so instead of naming a target that does
/// not exist. Oracle: `kernels/` ships gb10 (12.1), hopper (9.0) and b200
/// (10.0) — nothing for a Turing 7.5.
#[test]
fn a_compute_capability_with_no_shipped_target_says_so() {
    assert_eq!(target_hint((9, 0)), Some("hopper"));
    assert_eq!(target_hint((12, 1)), Some("gb10"));
    assert_eq!(target_hint((10, 0)), Some("b200"));
    assert_eq!(target_hint((7, 5)), None);
    let msg = ptx_arch_runs_on_device("sm_121f", (7, 5))
        .expect_err("mismatch expected")
        .to_string();
    assert!(msg.contains("no shipped target"), "{msg}");
    assert!(!msg.contains("ATLAS_TARGET_HW="), "{msg}");
}
