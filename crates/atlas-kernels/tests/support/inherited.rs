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
    /// The `common/` file names this target owns as REAL SOURCES rather than
    /// inheriting from gb10 — `[kernels] overrides` in its HARDWARE.toml.
    ///
    /// Per-target because the list is a claim about THIS hardware's tuning,
    /// even though both current targets happen to declare the same four: B200
    /// reaches them by symlinking Hopper's copies (ordinary datacentre MMA,
    /// nothing sm_90a about it), which is a sharing decision that could change
    /// the day a B200 receipt calls for something different.
    pub overrides: &'static [&'static str],
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

/// The Hopper-tuned sources that are NOT in `kernels/gb10/common` at all.
///
/// Maintainer rule, 2026-09-11 (tbraun96): "symlinks are fine provided the
/// pointed-to gb10 file is not edited when iterating on Hopper; Hopper-tuned
/// kernels must be real files under `kernels/hopper/`." These four exist for
/// H100 decode performance, are reached only under a `[defaults]` lever that
/// no GB10 target sets, and were moved out of gb10's tree so a GB10 build does
/// not compile them.
pub const HOPPER_TUNED: &[&str] = &[
    "dense_gemm_m16_bf16.cu",
    "fp8_scale_transpose.cu",
    "w8a16_gemm_m16.cu",
    "w8a16_gemv_ncol.cu",
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
        overrides: HOPPER_TUNED,
        blockscale_rejection: "cvt with .e2m1x2",
    },
    Inherited {
        hw: "b200",
        arch: "sm_100a",
        cc: "10.0",
        provenance: "B200 target: kernel set inherited from gb10 via symlink",
        models: P0_MODELS,
        overrides: HOPPER_TUNED,
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

/// `[kernels] overrides` from `kernels/<hw>/HARDWARE.toml` — the file names in
/// this target's `common/` that are NOT inherited from gb10.
///
/// The SSOT for "which kernels does this target tune for itself", read by the
/// mirror check here and reported by `scripts/check_kernel_shadows.py`. Empty
/// (and absent from the file) for a target that inherits everything.
pub fn kernel_overrides(hw: &str) -> std::collections::BTreeSet<String> {
    hardware_toml(hw)
        .get("kernels")
        .and_then(|k| k.get("overrides"))
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|v| {
                    v.as_str()
                        .unwrap_or_else(|| {
                            panic!("kernels/{hw}: [kernels] overrides entries must be strings")
                        })
                        .to_string()
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn hardware_toml(hw: &str) -> toml::Value {
    let path = hw_dir(hw).join("HARDWARE.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    toml::from_str(&text).unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()))
}
