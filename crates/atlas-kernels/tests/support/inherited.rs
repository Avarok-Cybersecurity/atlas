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
    /// The ptxas rejection this hardware answers by defining
    /// `ATLAS_NO_WARP_BLOCKSCALE_MMA` — the arch-specific half of the reason,
    /// which the MODEL.toml entries must cite. Per-target because the two
    /// architectures reject the W4A4 region for DIFFERENT reasons.
    pub blockscale_rejection: &'static str,
}

impl Inherited {
    /// The `common/` entries this target OWNS rather than inherits — read
    /// from its own `HARDWARE.toml` `[kernels] overrides`, never from a list
    /// kept here.
    ///
    /// A METHOD, not a field, and that is the whole point. The declaration
    /// would otherwise arrive twice — once as `[kernels] overrides` in the
    /// TOML that `scripts/check_kernel_shadows.py` and the build read, once as
    /// a Rust constant these tests read — and two spellings of "which kernels
    /// does this target own" is exactly how a file comes to be declared in one
    /// and forgotten in the other. The TOML wins because it is the one the
    /// non-Rust consumers can read.
    ///
    /// Two SHAPES live in the one list, distinguished by whether the oracle
    /// has the same name:
    ///
    /// * an OVERRIDE replaces a gb10 namesake — same entry points, this
    ///   hardware's instruction selection, and gb10's own file left untouched
    ///   because other targets compile it;
    /// * an ADDITION has a stem gb10 does not have at all, and must bring
    ///   entry points gb10 does not declare (otherwise one target would
    ///   compile two definitions of one kernel name).
    ///
    /// EMPTY for every target in this tree today — hopper and b200 inherit
    /// everything. The mechanism is here so that a PR adding a tuned kernel
    /// declares it in the file the build, the shadow checker and these tests
    /// all read, instead of dropping a regular file into a mirror where
    /// nothing can tell it from a silent fork.
    pub fn owned(&self) -> std::collections::BTreeSet<String> {
        kernel_overrides(self.hw)
    }
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
        blockscale_rejection: "cvt with .e2m1x2",
    },
    Inherited {
        hw: "b200",
        arch: "sm_100a",
        cc: "10.0",
        provenance: "B200 target: kernel set inherited from gb10 via symlink",
        models: P0_MODELS,
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
/// mirror check here, counted by `build_summary::count_declared_overrides` for
/// the build line, and reported by `scripts/check_kernel_shadows.py`. Empty
/// (and absent from the file) for a target that inherits everything, which is
/// every target today.
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
