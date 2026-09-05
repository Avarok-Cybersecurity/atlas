// SPDX-License-Identifier: AGPL-3.0-only

//! Does the PTX this binary carries run on the GPU it was handed?
//!
//! Atlas compiles ONE SM architecture per build — `kernels/<hw>/HARDWARE.toml`
//! `[hardware].arch` picks it, there is no fatbin and no multi-`-gencode`. So a
//! binary and a GPU can simply disagree, and until this module existed nothing
//! checked: the mismatch surfaced as an opaque driver failure inside
//! `cuModuleLoadData` (`CUDA_ERROR_NO_BINARY_FOR_GPU` /
//! `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`) that names neither the arch we built
//! nor the card in the box.
//!
//! Pure and dependency-free on purpose: the rules are a property of NVIDIA's
//! PTX ABI, not of any backend, so they belong where both the CUDA preflight
//! and the `--check-kernels` reporter can reach them without a GPU.

/// The suffix on an `sm_XY…` architecture string, which is what decides how
/// far the compiled code travels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmSuffix {
    /// No suffix — portable PTX. JIT-compiles on any device with a compute
    /// capability at or above the compiled one.
    None,
    /// `a` — architecture-specific. Emits instructions that exist on exactly
    /// one architecture (Hopper `wgmma`, its TMA descriptors), so it is never
    /// forward-compatible.
    Arch,
    /// `f` — family-specific, added in CUDA 12.9. Runs on devices of the same
    /// major family at or above the compiled compute capability.
    Family,
}

/// A parsed `sm_XY[a|f]` architecture string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmArch {
    /// Compute-capability major (the `12` of `sm_121f`).
    pub major: u32,
    /// Compute-capability minor (the `1` of `sm_121f`).
    pub minor: u32,
    /// Which compatibility rule this arch obeys.
    pub suffix: SmSuffix,
}

/// Parse an NVIDIA `sm_XY[a|f]` architecture string.
///
/// Returns `None` for anything that is not an NVIDIA SM arch — `gfx1151`
/// (SCALE/HIP), `metal3.1`, or junk. `None` is therefore the "not an NVIDIA
/// target" marker callers test against.
///
/// The digits split NVIDIA's way: the LAST digit is the minor version, the
/// rest is the major (`sm_90` = 9.0, `sm_100` = 10.0, `sm_121` = 12.1).
pub fn parse_sm_arch(arch: &str) -> Option<SmArch> {
    let rest = arch.strip_prefix("sm_")?;
    let (digits, suffix) = match rest.as_bytes().last()? {
        b'a' => (&rest[..rest.len() - 1], SmSuffix::Arch),
        b'f' => (&rest[..rest.len() - 1], SmSuffix::Family),
        _ => (rest, SmSuffix::None),
    };
    // Two digits minimum: one for the major, one for the minor. `sm_9` is not
    // a thing NVIDIA emits, and guessing at it would invent a compatibility
    // verdict from a typo.
    if digits.len() < 2 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (major, minor) = digits.split_at(digits.len() - 1);
    Some(SmArch {
        major: major.parse().ok()?,
        minor: minor.parse().ok()?,
        suffix,
    })
}

/// Which `kernels/<hw>/` target ships for a device compute capability.
///
/// Deliberately tiny and explicit — it exists so the mismatch message can tell
/// an operator what to rebuild instead of leaving them to guess. `None` means
/// Atlas ships nothing for that GPU, and is the honest answer for every CC not
/// listed: naming a target that cannot run either would send someone to
/// rebuild an image that fails the same way (SM 10.3 Blackwell Ultra against
/// the 10.0 `sm_100a` build is the live example).
///
/// HAND-MAINTAINED, and deliberately so, even though
/// `kernels/<hw>/HARDWARE.toml` already declares `compute_capability` for each
/// of these. Deriving it would mean either a build script that reads the
/// kernels tree — atlas-core is a leaf crate with no build script and no TOML
/// parser, and this function has to work inside a shipped binary that carries
/// no `kernels/` at all — or baking the table in at build time, which trades a
/// three-line table for a code generator. The gap that choice leaves is that
/// the two can silently disagree, and
/// `atlas-kernels/tests/target_hints.rs` closes it: it reads every
/// `vendor = "nvidia"` HARDWARE.toml and asserts this function maps its
/// declared `compute_capability` back to its own directory name.
pub fn target_hint(device_cc: (u32, u32)) -> Option<&'static str> {
    match device_cc {
        (9, 0) => Some("hopper"),
        // Blackwell datacentre. NOT (10, 3): B300/GB300 are `sm_103a`, a
        // separate arch-specific target that does not exist in `kernels/`.
        (10, 0) => Some("b200"),
        (12, 1) => Some("gb10"),
        _ => None,
    }
}

/// The compiled kernels cannot run on the device in front of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchMismatch {
    /// The arch string the kernels were compiled for, verbatim.
    pub compiled_arch: String,
    /// The parsed form of `compiled_arch`, so the message can say WHY.
    pub compiled: SmArch,
    /// `(major, minor)` compute capability the driver reports for this device.
    pub device_cc: (u32, u32),
}

impl std::fmt::Display for ArchMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (major, minor) = self.device_cc;
        let arch = &self.compiled_arch;
        let reason = match self.compiled.suffix {
            SmSuffix::None => format!(
                "portable PTX for {arch} needs compute capability {}.{} or later",
                self.compiled.major, self.compiled.minor
            ),
            SmSuffix::Arch => format!(
                "architecture-specific PTX ({arch}) runs only on compute capability {}.{}",
                self.compiled.major, self.compiled.minor
            ),
            SmSuffix::Family => format!(
                "family-specific PTX ({arch}) runs only on compute capability {}.{} or later \
                 within the {}.x family",
                self.compiled.major, self.compiled.minor, self.compiled.major
            ),
        };
        let fix = match target_hint(self.device_cc) {
            Some(hw) => format!(
                "rebuild with ATLAS_TARGET_HW={hw} (kernels/{hw}/HARDWARE.toml arch must match \
                 this GPU) or use the image built for this GPU"
            ),
            None => format!(
                "no shipped target matches compute capability {major}.{minor} \
                 (kernels/<hw>/HARDWARE.toml arch must match this GPU) — \
                 use the image built for this GPU"
            ),
        };
        write!(
            f,
            "kernels compiled for {arch} cannot run on this GPU \
             (compute capability {major}.{minor}): {reason}; fix: {fix}"
        )
    }
}

impl std::error::Error for ArchMismatch {}

/// Can PTX compiled for `compiled_arch` run on a device of `device_cc`?
///
/// ORACLE: NVIDIA CUDA C++ Programming Guide, "Application Compatibility" →
/// *PTX Compatibility*, plus NVIDIA's CUDA 12.9 announcement of family-specific
/// features (developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-
/// introduce-family-specific-architecture-features/, re-read 2026-09-05):
/// `compute_100f` "is compatible with all CC 10.x devices (sm_100, sm_103)",
/// while the `a` suffix "is not forward-compatible with any future GPU
/// architecture". Exactly three rules:
///
/// 1. plain `sm_XY` runs on any device with CC >= X.Y (JIT forward-compat);
/// 2. `sm_XYa` runs ONLY on CC == X.Y;
/// 3. `sm_XYf` runs on the same major family with CC >= X.Y.
///
/// A `compiled_arch` that is not an NVIDIA SM arch (`gfx1151`, `metal3.1`)
/// returns `Ok(())`: a CUDA compute capability says nothing about it, and
/// inventing a verdict would fail every AMD and Apple build. Callers that need
/// to know use [`parse_sm_arch`], whose `None` is the marker.
pub fn ptx_arch_runs_on_device(
    compiled_arch: &str,
    device_cc: (u32, u32),
) -> Result<(), ArchMismatch> {
    let Some(compiled) = parse_sm_arch(compiled_arch) else {
        return Ok(());
    };
    let compiled_cc = (compiled.major, compiled.minor);
    let runs = match compiled.suffix {
        SmSuffix::None => device_cc >= compiled_cc,
        SmSuffix::Arch => device_cc == compiled_cc,
        SmSuffix::Family => device_cc.0 == compiled.major && device_cc >= compiled_cc,
    };
    if runs {
        return Ok(());
    }
    Err(ArchMismatch {
        compiled_arch: compiled_arch.to_string(),
        compiled,
        device_cc,
    })
}

#[cfg(test)]
#[path = "arch_tests.rs"]
mod tests;
