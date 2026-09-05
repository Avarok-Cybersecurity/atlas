// SPDX-License-Identifier: AGPL-3.0-only

//! The `ATLAS_NO_WARP_BLOCKSCALE_MMA` opt-out on the targets that inherit
//! `kernels/gb10`'s kernels, and the `[expected_absent]` declarations that
//! answer it.
//!
//! Split out of `inherited_targets.rs` at the 500-LoC cap. That binary pins the
//! tree as MIRRORED — every inherited file is gb10's file, reachable. This one
//! pins the single place the two trees deliberately DIVERGE from gb10: ptxas
//! rejects the warp-block-scale W4A4 region on both sm_90a and sm_100a, so both
//! targets compile it out, and each must then declare the two entry points it
//! loses with a reason naming its OWN rejection. gb10, whose PTX may not move,
//! must declare neither.
//!
//! Parametrised over [`INHERITED`] from `support/inherited.rs` — the same
//! declaration `inherited_targets.rs` reads, so a target added to one binary
//! cannot be missed by the other.
//!
//! `cargo test` runs GPU-free with `ATLAS_SKIP_BUILD=1`, where `build.rs`
//! returns before target resolution ever happens, so without this file nothing
//! in CI looks at either tree at all.

#[path = "support/inherited.rs"]
mod inherited;

use inherited::{INHERITED, gb10_dir, hardware_toml, hw_dir};

/// The define that compiles the warp-block-scale W4A4 path out.
const GUARD_FLAG: &str = "-DATLAS_NO_WARP_BLOCKSCALE_MMA";

/// The model whose kernel set contains that path, and the two entry points it
/// loses. ORACLE for the pair: the `extern "C" __global__` declarations inside
/// the `#ifndef` region of the gb10 source, checked below rather than trusted.
const GUARDED_MODEL: &str = "qwen3.6-35b-a3b";
const GUARDED_MODULE: &str = "moe_w4a16";
const GUARDED_KERNELS: &[&str] = &[
    "moe_w4a16_fused_gate_up_t_k64_fp4",
    "moe_w4a16_down_t_k64_fp4",
];

/// ORACLE: the two ptxas rejections measured on Spark 1 (CUDA 13.0.88),
/// receipts under `docs/campaigns/hopper-atlas-vs-vllm-2026-09/receipts/`.
/// Both targets define the guard; gb10 must NOT, because its PTX may not move.
#[test]
fn both_inherited_targets_compile_out_the_warp_block_scale_path() {
    for t in INHERITED {
        let flags = hardware_toml(t.hw)
            .get("build")
            .and_then(|b| b.get("extra_nvcc_flags"))
            .and_then(|f| f.as_array())
            .map(|a| {
                a.iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        assert!(
            flags.iter().any(|f| f == GUARD_FLAG),
            "kernels/{}/HARDWARE.toml must define {GUARD_FLAG}: ptxas rejects \
             the W4A4 region here with {:?}",
            t.hw,
            t.blockscale_rejection
        );
    }
    let gb10: toml::Value =
        toml::from_str(&std::fs::read_to_string(gb10_dir().join("HARDWARE.toml")).unwrap())
            .unwrap();
    assert!(
        gb10.get("build").is_none(),
        "kernels/gb10/HARDWARE.toml must add no flags — the W4A4 path is \
         compiled IN on GB10 and its PTX may not move"
    );
}

/// The kernels the define removes are exactly the `__global__` entry points
/// inside the guarded region of the gb10 source. Derived, not asserted from
/// memory: a third W4A4 entry point added inside the guard and not declared
/// below would otherwise reach the boot gate on Hopper as a required
/// unresolved lookup, which is the failure `[expected_absent]` exists to make
/// impossible.
#[test]
fn the_declared_absences_are_the_entry_points_the_define_removes() {
    let src = gb10_dir()
        .join(GUARDED_MODEL)
        .join("nvfp4/moe_w4a16_grouped_gemm.cu");
    let text = std::fs::read_to_string(&src).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
    let (_, guarded) = text
        .split_once("#ifndef ATLAS_NO_WARP_BLOCKSCALE_MMA")
        .unwrap_or_else(|| panic!("{}: no guard", src.display()));
    let mut inside: Vec<&str> = guarded
        .lines()
        .filter_map(|l| l.strip_prefix("extern \"C\" __global__ void "))
        .filter_map(|l| l.split('(').next())
        .collect();
    inside.sort_unstable();
    let mut expected: Vec<&str> = GUARDED_KERNELS.to_vec();
    expected.sort_unstable();
    assert_eq!(
        inside,
        expected,
        "{}: the entry points inside the guard are not the ones both \
         MODEL.tomls declare expected-absent",
        src.display()
    );
}

/// Both targets declare both kernels in `[expected_absent.moe_w4a16]`, each
/// with a reason that names THAT architecture's ptxas rejection.
///
/// A bare declaration would silence the boot gate without recording why, and
/// the two reasons are genuinely different: Hopper has no NVFP4 datapath at
/// all, datacentre Blackwell has one and reaches it through tcgen05. A reader
/// who cannot tell those apart cannot tell which target a tcgen05 port would
/// fix.
#[test]
fn both_inherited_targets_declare_the_w4a4_kernels_expected_absent() {
    for t in INHERITED {
        let path = hw_dir(t.hw).join(GUARDED_MODEL).join("MODEL.toml");
        let toml: toml::Value = toml::from_str(&std::fs::read_to_string(&path).unwrap())
            .unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()));
        let table = toml
            .get("expected_absent")
            .and_then(|e| e.get(GUARDED_MODULE))
            .and_then(|m| m.as_table())
            .unwrap_or_else(|| panic!("{}: no [expected_absent.{GUARDED_MODULE}]", path.display()));
        for kernel in GUARDED_KERNELS {
            let reason = table
                .get(*kernel)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{}: {kernel} is not declared", path.display()));
            assert!(
                reason.contains(t.blockscale_rejection),
                "{}: {kernel}'s reason does not name this architecture's ptxas \
                 rejection {:?}:\n{reason}",
                path.display(),
                t.blockscale_rejection
            );
            assert!(
                reason.contains("ATLAS_NO_WARP_BLOCKSCALE_MMA"),
                "{}: {kernel}'s reason does not say what compiles it out",
                path.display()
            );
        }
    }
}

/// gb10 declares NEITHER kernel absent: it compiles both, and a declaration
/// there would mean the guard had leaked onto the target it must not touch.
#[test]
fn gb10_declares_neither_w4a4_kernel_absent() {
    let path = gb10_dir().join(GUARDED_MODEL).join("MODEL.toml");
    let toml: toml::Value = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let table = toml
        .get("expected_absent")
        .and_then(|e| e.get(GUARDED_MODULE))
        .and_then(|m| m.as_table());
    for kernel in GUARDED_KERNELS {
        assert!(
            table.map(|t| t.get(*kernel).is_none()).unwrap_or(true),
            "{}: {kernel} is compiled on GB10 and must not be declared absent",
            path.display()
        );
    }
}
