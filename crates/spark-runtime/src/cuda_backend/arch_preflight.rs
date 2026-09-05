// SPDX-License-Identifier: AGPL-3.0-only

//! Refuse to load kernels the GPU cannot run, BEFORE the driver does it badly.
//!
//! Atlas compiles one SM architecture per build, and the driver's answer to a
//! mismatch is `CUDA_ERROR_NO_BINARY_FOR_GPU` (or
//! `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`) raised inside `cuModuleLoadData` — an
//! error that names neither the arch in the binary nor the card in the box. An
//! operator who boots the published gb10 image on an H100 gets that, and
//! nothing to act on.
//!
//! So this runs first: two `cuDeviceGetAttribute` calls, the pure rule from
//! [`atlas_core::arch`], and a message that names both sides. The rule itself
//! lives in atlas-core because `--check-kernels` reports it too.
//!
//! No new FFI: `cuCtxGetDevice` and `cuDeviceGetAttribute` were already
//! declared for the SM-count query.

use anyhow::{Result, bail};

use super::{cuCtxGetDevice, cuDeviceGetAttribute};

/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` — CUDA driver API enum 75.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: u32 = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR` — CUDA driver API enum 76.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: u32 = 76;

/// One `CUdevice_attribute` on `dev`, or the driver status that refused it.
fn device_attribute(attrib: u32, dev: i32) -> Result<i32> {
    let mut value: i32 = 0;
    let status = unsafe { cuDeviceGetAttribute(&mut value, attrib, dev) };
    if status != 0 {
        bail!("cuDeviceGetAttribute({attrib}) failed: status {status}");
    }
    Ok(value)
}

/// `(major, minor)` compute capability of the calling context's device.
///
/// Requires a current CUDA context, exactly like `sm_count_cu` next door.
/// Fails loudly rather than guessing: a fabricated compute capability would
/// turn this preflight into a rubber stamp.
pub fn device_compute_capability() -> Result<(u32, u32)> {
    let mut dev: i32 = 0;
    let status = unsafe { cuCtxGetDevice(&mut dev) };
    if status != 0 {
        bail!("cuCtxGetDevice failed: status {status}");
    }
    let major = device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)?;
    let minor = device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)?;
    if major <= 0 {
        bail!("driver reported compute capability {major}.{minor} on device {dev}");
    }
    Ok((major as u32, minor as u32))
}

/// The verdict, without touching a GPU: `Ok(line to log)` or the mismatch.
///
/// Split out so the decision is testable on a host with no CUDA at all, which
/// is every machine CI runs on.
pub fn check_arch(compiled_arch: &str, device_cc: (u32, u32)) -> Result<String> {
    if let Err(mismatch) = atlas_core::arch::ptx_arch_runs_on_device(compiled_arch, device_cc) {
        bail!("{mismatch}");
    }
    Ok(format!(
        "device CC {}.{}, kernels built for {compiled_arch}",
        device_cc.0, device_cc.1
    ))
}

/// Which architecture string a resolved target's preflight must judge.
///
/// A `TargetPtxSet` carries two readings of one `[hardware].arch`
/// declaration, and only one of them can answer this question:
///
/// * `target.arch` is the BASE SM (`sm_90`, `sm_121`) — the identity the
///   registry, `KernelTarget`'s constants and every gate baseline are keyed
///   by. Its feature suffix has been stripped, so `sm_90a` arrives as plain
///   `sm_90`, which the forward-compat rule says runs on any CC >= 9.0.
/// * `ptx_arch` is the declaration VERBATIM (`sm_90a`, `sm_121f`) — what nvcc
///   was handed, suffix and all. The suffix IS the compatibility rule.
///
/// Passing the base SM here is not a slightly weaker check, it is the wrong
/// one: Hopper-only PTX would pass on a B200 (CC 10.0) or a GB10 (12.1) and
/// then fail inside `cuModuleLoadData` — the driver error with no useful
/// nouns in it that this whole module exists to pre-empt.
///
/// `None` when the target records no architecture, which the caller warns
/// about and skips rather than treating as a pass.
pub fn preflight_arch(ptx_set: &atlas_kernels::TargetPtxSet) -> Option<&'static str> {
    Some(ptx_set.ptx_arch).filter(|a| !a.is_empty())
}

/// Fail fast if this binary's kernels cannot run on GPU `ordinal`.
///
/// Call this BEFORE constructing the backend: `AtlasCudaBackend::new` loads
/// every PTX module, and the point is to answer before the driver does.
///
/// `compiled_arch` is `None` when the build recorded no architecture — the
/// `ATLAS_SKIP_BUILD=1` stub registry compiles nothing and can attest to
/// nothing. That is warned and skipped, never treated as a pass: a check with
/// no input has no opinion, and inventing one would make the stub build claim
/// hardware compatibility it never tested.
pub fn preflight_device_arch(ordinal: usize, compiled_arch: Option<&str>) -> Result<()> {
    let Some(compiled_arch) = compiled_arch else {
        tracing::warn!(
            "this build recorded no kernel architecture, so the GPU compute-capability \
             preflight is skipped — expected under ATLAS_SKIP_BUILD=1, a defect otherwise"
        );
        return Ok(());
    };
    // Bind the process CUDA host first: `cuCtxGetDevice` needs a current
    // context, and this is upstream of every `cuModuleLoadData`. `host` is a
    // `OnceLock` on the same ordinal the backend will ask for, so the backend
    // reuses this context rather than creating a second one.
    atlas_core::cuda_host::host(ordinal).map_err(|e| anyhow::anyhow!("{e}"))?;
    let device_cc = device_compute_capability()?;
    tracing::info!("{}", check_arch(compiled_arch, device_cc)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_arch, preflight_arch, preflight_device_arch};
    use atlas_core::target::KernelTarget;
    use atlas_kernels::{ModelBehavior, SamplingPresets, TargetPtxSet};

    /// A `TargetPtxSet` shaped exactly as `build_codegen.rs` emits one for
    /// `kernels/hopper`: `KernelTarget.arch` is the base SM the registry is
    /// keyed by, `ptx_arch` is the `[hardware].arch` nvcc was handed.
    fn a_hopper_target(ptx_arch: &'static str) -> TargetPtxSet {
        TargetPtxSet {
            target: KernelTarget {
                arch: "sm_90",
                model: "nemotron-super-120b-a12b",
                quant: "nvfp4",
            },
            ptx_arch,
            modules: Vec::new(),
            sampling: SamplingPresets::default(),
            behavior: ModelBehavior::default(),
            model_type_matches: Vec::new(),
            match_names: &[],
            dflash: None,
            shadowed_dropped: &[],
            expected_absent: &[],
        }
    }

    /// ★ THE DEFECT, pinned. `KernelTarget.arch` records the base SM, so a
    /// hopper build reaches this module as `sm_90` — plain PTX, which the
    /// forward-compat rule says runs on any CC >= 9.0. A B200 (CC 10.0) or a
    /// GB10 (12.1) would therefore PASS the preflight and then fail inside
    /// `cuModuleLoadData`, which is precisely the driver error this preflight
    /// exists to replace.
    ///
    /// Oracle: `kernels/hopper/HARDWARE.toml` declares `arch = "sm_90a"`, and
    /// the NVIDIA CUDA C++ Programming Guide's *PTX Compatibility* rules make
    /// an `a`-suffixed arch runnable on CC 9.0 and nothing else. So the
    /// preflight must judge `ptx_arch`, and the pick is what this asserts —
    /// `check_arch` itself was already correct about `sm_90a`; nothing called
    /// it with `sm_90a`.
    #[test]
    fn the_preflight_judges_the_verbatim_arch_not_the_stripped_base_sm() {
        let hopper = a_hopper_target("sm_90a");
        assert_eq!(
            preflight_arch(&hopper),
            Some("sm_90a"),
            "the preflight must be handed the arch nvcc compiled for"
        );
        // The negative the whole slice is for: Hopper PTX on Blackwell
        // datacenter silicon.
        let err = check_arch(
            preflight_arch(&hopper).expect("hopper records an arch"),
            (10, 0),
        )
        .expect_err("sm_90a cannot load on CC 10.0");
        let msg = format!("{err}");
        assert!(msg.contains("sm_90a"), "{msg}");
        assert!(msg.contains("compute capability 10.0"), "{msg}");
        // …and the base SM, which is what USED to be passed, is waved through.
        // Asserted so the two readings are visibly not interchangeable rather
        // than merely documented as such.
        assert!(
            check_arch(hopper.target.arch, (10, 0)).is_ok(),
            "sm_90 is plain PTX and passes on CC 10.0 — that is the bug, not a \
             property to rely on"
        );
    }

    /// A build that compiled nothing records no arch, and the skip branch must
    /// still fire through the selector.
    ///
    /// Oracle: `crates/atlas-kernels/build.rs` under `ATLAS_SKIP_BUILD=1`
    /// writes a stub whose `all_ptx_sets()` is empty, so nothing carries an
    /// arch at all; an empty `ptx_arch` is the same statement reaching a
    /// consumer that does hold a set.
    #[test]
    fn a_target_that_records_no_arch_selects_nothing_to_check() {
        let stub = a_hopper_target("");
        assert_eq!(preflight_arch(&stub), None);
        // The whole chain, as the serve phase runs it: an empty `ptx_arch`
        // reaches `preflight_device_arch` as `None`, which warns and returns
        // WITHOUT touching CUDA — so this passes on the GPU-free runner.
        preflight_device_arch(0, preflight_arch(&stub)).expect("a stub build has nothing to check");
    }

    /// Oracle: `kernels/gb10/HARDWARE.toml` declares `arch = "sm_121f"` and
    /// `compute_capability = "12.1"` — the shipped pairing must pass, and the
    /// line it logs must name both halves so a support ticket can quote it.
    #[test]
    fn a_matching_device_logs_both_the_device_and_the_compiled_arch() {
        let line = check_arch("sm_121f", (12, 1)).expect("gb10 kernels run on a gb10");
        assert_eq!(line, "device CC 12.1, kernels built for sm_121f");
    }

    /// Oracle: an H100 is compute capability 9.0 and `sm_90a` is
    /// architecture-specific to it. This is the pairing the Hopper target
    /// exists to serve.
    #[test]
    fn hopper_kernels_pass_on_a_hopper_device() {
        let line = check_arch("sm_90a", (9, 0)).expect("hopper kernels run on hopper");
        assert_eq!(line, "device CC 9.0, kernels built for sm_90a");
    }

    /// The bring-up failure this module exists to intercept: the published
    /// gb10 image booted on an H100. The error must carry the operator-facing
    /// message rather than a driver status code.
    #[test]
    fn the_gb10_image_on_a_hopper_device_fails_with_the_operator_message() {
        let err = check_arch("sm_121f", (9, 0)).expect_err("sm_121f cannot load on CC 9.0");
        let msg = format!("{err}");
        assert!(msg.contains("sm_121f"), "{msg}");
        assert!(msg.contains("compute capability 9.0"), "{msg}");
        assert!(msg.contains("ATLAS_TARGET_HW=hopper"), "{msg}");
    }

    /// A build that compiled nothing has nothing to check.
    ///
    /// Oracle: `crates/atlas-kernels/build.rs` writes a stub `target_ptx.rs`
    /// under `ATLAS_SKIP_BUILD=1` whose `all_ptx_sets()` is empty — no arch is
    /// recorded anywhere. This branch must return before it touches CUDA, or
    /// every GPU-free `cargo test` host would fail it.
    #[test]
    fn a_build_that_recorded_no_arch_skips_the_check_without_a_gpu() {
        preflight_device_arch(0, None).expect("a stub build has nothing to check");
    }
}
