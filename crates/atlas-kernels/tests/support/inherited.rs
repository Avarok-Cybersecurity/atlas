// SPDX-License-Identifier: AGPL-3.0-only

//! WHICH hardware sets inherit `kernels/gb10`'s kernels, and where their files
//! live on disk.
//!
//! Shared by `tests/inherited_targets.rs` and
//! `tests/inherited_targets_w4a4.rs`, which pin two different properties of the
//! same two trees and must agree on what those trees ARE: a second copy of
//! [`INHERITED`] would let one binary keep testing a target the other had
//! already been told about. Lives under `tests/support/` so cargo does not pick
//! it up as a test target of its own — a subdirectory is not auto-discovered, a
//! `tests/*.rs` is — the same reason `mirror.rs` is here.
//!
//! Each including binary reads the subset of this file it needs (the W4A4
//! binary asks only for `hw` and `blockscale_rejection`), so the unread
//! remainder is dead code in exactly one of the two — allowed here rather than
//! split further, because the point of the file is that ONE declaration
//! describes each target.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// One hardware set that inherits gb10's kernels.
pub struct Inherited {
    /// `kernels/<hw>` directory name.
    pub hw: &'static str,
    /// `[hardware].arch`, verbatim — what reaches `nvcc -arch=`.
    pub arch: &'static str,
    /// `[hardware].compute_capability`.
    pub cc: &'static str,
    /// The opening line every copied MODEL.toml must carry, naming where the
    /// kernels came from. Per-target because it names the target.
    pub provenance: &'static str,
    /// The campaign's declared P0 model set for this hardware.
    pub models: &'static [&'static str],
    /// `common/` entries this hardware set TUNES FOR ITSELF: a real file that
    /// overrides its gb10 namesake, plus any header only that file needs.
    /// Maintainer rule, 2026-09-11 — a Hopper-tuned kernel must not edit the
    /// gb10 source; gb10 keeps its kernel and the hardware target overrides
    /// it. Every name here is checked BOTH ways by `mirror::mirror_faults`:
    /// it must exist and must be a regular file, and a name left off the list
    /// that stops being a symlink is still reported as an undeclared fork.
    pub owned: &'static [&'static str],
    /// The ptxas rejection this hardware answers by defining
    /// `ATLAS_NO_WARP_BLOCKSCALE_MMA` — the arch-specific half of the reason,
    /// which the MODEL.toml entries must cite. Per-target because the two
    /// architectures reject the W4A4 region for DIFFERENT reasons.
    pub blockscale_rejection: &'static str,
}

/// The five P0 models shared by the Hopper/B200 campaign.
pub const P0_MODELS: &[&str] = &[
    "deepseek-v4-flash",
    "nemotron-3-nano-30b-a3b",
    "nemotron-super-120b-a12b",
    "qwen3-next-80b-a3b",
    "qwen3.6-35b-a3b",
];

/// Hopper additionally carries the PRD section 16 first paid 27B cell and
/// the same-hardware source target its kernel_source redirect requires.
pub const HOPPER_MODELS: &[&str] = &[
    "deepseek-v4-flash",
    "nemotron-3-nano-30b-a3b",
    "nemotron-super-120b-a12b",
    "qwen3-next-80b-a3b",
    "qwen3.6-27b",
    "qwen3.6-35b-a3b",
    "qwen3.8-27b",
];

/// `kernels/hopper/common` entries that are Hopper's own, not gb10's.
///
/// The W8A16 M=1 decode GEMV family (#928). On an H100 these three entry
/// points are 71% of the C=1 decode step (nsys, 1xH100, Qwen/Qwen3.8-27B-FP8,
/// 2026-09-11 round 10) and the gb10 kernels are LSU-bound on the E4M3 LUT
/// gather rather than bandwidth-bound; the reasoning, the arithmetic and the
/// bit-identity argument are in `kernels/hopper/common/w8a16_gemv_hopper.cuh`.
/// The `.cuh` has no gb10 counterpart by design — it is the shared inner loop
/// of the two `.cu` overrides and nothing else includes it.
pub const HOPPER_OWNED_COMMON: &[&str] = &[
    "w8a16_gemv.cu",
    "w8a16_gemv_fused.cu",
    "w8a16_gemv_hopper.cuh",
];

/// Every hardware set whose kernels are gb10's, reached by symlink.
///
/// ORACLE for the arch strings: NVIDIA's own SM numbering. H100 and H200 are
/// both SM 9.0; B200 and GB200 are SM 10.0. The `a` suffix is nvcc's
/// arch-specific spelling — it opts the target into wgmma/TMA on Hopper and
/// tcgen05/native-NVFP4 on Blackwell datacentre, and it makes the PTX
/// non-forward-compatible, which is correct for a per-architecture target.
pub const INHERITED: &[Inherited] = &[
    Inherited {
        hw: "hopper",
        arch: "sm_90a",
        cc: "9.0",
        provenance: "Hopper target: kernel set inherited from gb10 via symlink",
        models: HOPPER_MODELS,
        owned: HOPPER_OWNED_COMMON,
        blockscale_rejection: "cvt with .e2m1x2",
    },
    Inherited {
        hw: "b200",
        arch: "sm_100a",
        cc: "10.0",
        provenance: "B200 target: kernel set inherited from gb10 via symlink",
        models: P0_MODELS,
        owned: &[],
        blockscale_rejection: "mma with block scale",
    },
];

pub fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/atlas-kernels is two levels below the workspace root")
        .join("kernels")
}

pub fn hw_dir(hw: &str) -> PathBuf {
    kernels_root().join(hw)
}

pub fn gb10_dir() -> PathBuf {
    kernels_root().join("gb10")
}

pub fn hardware_toml(hw: &str) -> toml::Value {
    let path = hw_dir(hw).join("HARDWARE.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    toml::from_str(&text).unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()))
}
