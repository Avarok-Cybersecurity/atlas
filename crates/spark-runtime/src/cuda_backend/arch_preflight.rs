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
//! The capability query is addressed BY ORDINAL (`cuDeviceGet`), not by "the
//! calling thread's current context" (`cuCtxGetDevice`). That is not a style
//! preference — see [`device_compute_capability_of`]. `cuDeviceGetAttribute`
//! and `cuCtxGetDevice` were already declared for the SM-count query;
//! `cuDeviceGet` is the one addition, alongside them.

use anyhow::{Result, bail};

use super::{cuCtxGetDevice, cuDeviceGet, cuDeviceGetAttribute};

/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` — CUDA driver API enum 75.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: u32 = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR` — CUDA driver API enum 76.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: u32 = 76;
/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` — CUDA driver API enum 16.
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: u32 = 16;

/// One `CUdevice_attribute` on `dev`, or the driver status that refused it.
fn device_attribute(attrib: u32, dev: i32) -> Result<i32> {
    let mut value: i32 = 0;
    let status = unsafe { cuDeviceGetAttribute(&mut value, attrib, dev) };
    if status != 0 {
        bail!("cuDeviceGetAttribute({attrib}) failed: status {status}");
    }
    Ok(value)
}

/// `(major, minor)` of an already-resolved `CUdevice`.
///
/// Fails loudly rather than guessing: a fabricated compute capability would
/// turn this preflight into a rubber stamp.
fn compute_capability_of_device(dev: i32) -> Result<(u32, u32)> {
    let major = device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)?;
    let minor = device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)?;
    if major <= 0 {
        bail!("driver reported compute capability {major}.{minor} on device {dev}");
    }
    Ok((major as u32, minor as u32))
}

/// `(major, minor)` compute capability of the calling context's device.
///
/// Requires a current CUDA context, exactly like `sm_count_cu` next door, so
/// it is only safe to call from a thread that has one. `--check-kernels` runs
/// it after the backend is up, which is such a thread. **The preflight does
/// not** — see [`device_compute_capability_of`].
pub fn device_compute_capability() -> Result<(u32, u32)> {
    let mut dev: i32 = 0;
    let status = unsafe { cuCtxGetDevice(&mut dev) };
    if status != 0 {
        bail!("cuCtxGetDevice failed: status {status}");
    }
    compute_capability_of_device(dev)
}

/// `(major, minor)` compute capability of GPU `ordinal`, with NO current
/// context required on the calling thread.
///
/// This exists because the context-addressed spelling above is wrong for a
/// preflight, and quietly so. `cuda_host::host(ordinal)` binds a context only
/// while it INITIALISES: once its `OnceLock` is populated it hands back an
/// `Arc` clone and touches no thread-current state. A TUI Library swap runs
/// the new load on a fresh `atlas-swap` thread while the previous model's
/// context was made current on the scheduler thread, so on the swap thread
/// `cuCtxGetDevice` has no context to read and returns
/// `CUDA_ERROR_INVALID_CONTEXT` — failing the requested load AND the attempt
/// to restore the old model, leaving the host with no model at all.
///
/// `cuDeviceGet` reads no thread-current state (NVIDIA's context API
/// documents the thread-current requirement as belonging to `cuCtx*`, not to
/// device enumeration), so the preflight needs no bind and cannot be made
/// wrong by which thread it runs on. `cuInit` is still a precondition, and the
/// `host(ordinal)` call in `preflight_device_arch_with` is what satisfies it.
pub fn device_compute_capability_of(ordinal: usize) -> Result<(u32, u32)> {
    let ordinal_i32 = i32::try_from(ordinal)
        .map_err(|_| anyhow::anyhow!("GPU ordinal {ordinal} does not fit a CUdevice ordinal"))?;
    let mut dev: i32 = 0;
    let status = unsafe { cuDeviceGet(&mut dev, ordinal_i32) };
    if status != 0 {
        bail!("cuDeviceGet(ordinal {ordinal}) failed: status {status}");
    }
    compute_capability_of_device(dev)
}

/// Streaming multiprocessors on GPU `ordinal`, with NO current context
/// required — addressed the same way, and for the same reason, as
/// [`device_compute_capability_of`].
pub fn device_sm_count_of(ordinal: usize) -> Result<u32> {
    let ordinal_i32 = i32::try_from(ordinal)
        .map_err(|_| anyhow::anyhow!("GPU ordinal {ordinal} does not fit a CUdevice ordinal"))?;
    let mut dev: i32 = 0;
    let status = unsafe { cuDeviceGet(&mut dev, ordinal_i32) };
    if status != 0 {
        bail!("cuDeviceGet(ordinal {ordinal}) failed: status {status}");
    }
    let count = device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)?;
    if count <= 0 {
        bail!("driver reported {count} multiprocessors on device {dev}");
    }
    Ok(count as u32)
}

/// Does the running device have the SM count this build's kernels were sized
/// for? `None` when it agrees, `Some(warning)` when it does not.
///
/// # Why this WARNS and does not fail
///
/// `kernels/<hw>/HARDWARE.toml` `[hardware] sm_count` feeds grid sizing —
/// today the paged-decode split-K split count (#928,
/// `atlas_kernels::attn_splitk`). A wrong value costs a suboptimal grid, never
/// a wrong answer: the split-K workspace is allocated from the SAME constant,
/// so the grid can never outrun its allocation, and the split partition is
/// clamped inside the kernel by each sequence's own length. Refusing to boot
/// on, say, an H100 PCIe with a different bin would be a regression dressed as
/// rigour.
///
/// It exists at all because the defect it guards was exactly a silent wrong
/// constant: `atlas_core::device::sm121::NUM_SMS = 48` compiled into an H100
/// build, where `paged_decode_attn_fp8` then ran 24 CTAs on 132 SMs for a
/// whole campaign. A one-line boot warning naming both numbers is what would
/// have caught it in round 1.
pub fn check_sm_count(device_sms: u32, declared_sms: u32) -> Option<String> {
    (device_sms != declared_sms).then(|| {
        format!(
            "this build's kernels are sized for {declared_sms} SMs              (kernels/{hw}/HARDWARE.toml [hardware] sm_count) but the device              reports {device_sms} — grid sizing that reads it (paged-decode              split-K) will be off; serving is unaffected",
            hw = atlas_kernels::TARGET_DEFAULTS.hw,
        )
    })
}

/// The verdict, without touching a GPU: `Ok(line to log)` or the mismatch.
///
/// Split out so the decision is testable on a host with no CUDA at all, which
/// is every machine CI runs on.
pub fn check_arch(compiled_arch: &str, device_cc: (u32, u32)) -> Result<String> {
    if let Err(mismatch) = atlas_core::arch::ptx_arch_runs_on_device(compiled_arch, device_cc) {
        // Keep the device facts typed so --check-kernels can report an early
        // refusal without parsing this error's human-readable message.
        return Err(mismatch.into());
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
    preflight_device_arch_with(ordinal, compiled_arch, &DriverDeviceQuery)
}

/// The two driver facts the preflight needs, behind a seam.
///
/// Not indirection for its own sake: the property that broke here — WHICH
/// THREAD each of the two runs on, and whether the second depends on the
/// first having run on that same thread — is invisible to any test that can
/// only call the real driver, and CI has no GPU to call it with. Behind this
/// trait the ordering contract is assertable on a bare host.
pub(crate) trait DeviceQuery {
    /// Initialise the process CUDA host on `ordinal` (`cuInit`, primary
    /// context). Idempotent, and — the whole point — binds a context to the
    /// CALLING thread on the first call only.
    fn init_host(&self, ordinal: usize) -> Result<()>;

    /// `(major, minor)` compute capability of GPU `ordinal`.
    ///
    /// Takes the ordinal, so an implementation CAN answer without a current
    /// context; [`DriverDeviceQuery`] is the one that does.
    fn compute_capability(&self, ordinal: usize) -> Result<(u32, u32)>;

    /// Streaming multiprocessors on GPU `ordinal`, for the [`check_sm_count`]
    /// cross-check.
    fn sm_count(&self, ordinal: usize) -> Result<u32>;
}

/// The production `DeviceQuery`: the process CUDA host, then `cuDeviceGet`.
pub(crate) struct DriverDeviceQuery;

impl DeviceQuery for DriverDeviceQuery {
    fn init_host(&self, ordinal: usize) -> Result<()> {
        atlas_core::cuda_host::host(ordinal).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    fn compute_capability(&self, ordinal: usize) -> Result<(u32, u32)> {
        device_compute_capability_of(ordinal)
    }

    fn sm_count(&self, ordinal: usize) -> Result<u32> {
        device_sm_count_of(ordinal)
    }
}

/// [`preflight_device_arch`] against an injected driver.
pub(crate) fn preflight_device_arch_with(
    ordinal: usize,
    compiled_arch: Option<&str>,
    query: &dyn DeviceQuery,
) -> Result<()> {
    let Some(compiled_arch) = compiled_arch else {
        tracing::warn!(
            "this build recorded no kernel architecture, so the GPU compute-capability \
             preflight is skipped — expected under ATLAS_SKIP_BUILD=1, a defect otherwise"
        );
        return Ok(());
    };
    // Initialise the process CUDA host first, for `cuInit` and so the backend
    // reuses this context rather than creating a second one — NOT to make a
    // context current, which on any thread after the first it does not do.
    // The capability query below is addressed by ordinal precisely so that
    // does not matter; see `device_compute_capability_of`.
    query.init_host(ordinal)?;
    let device_cc = query.compute_capability(ordinal)?;
    tracing::info!("{}", check_arch(compiled_arch, device_cc)?);
    // The SM-count cross-check (#928). A driver that will not answer is not a
    // reason to refuse a boot the arch check already passed — it is one fewer
    // cross-check, logged as such.
    match query.sm_count(ordinal) {
        Ok(device_sms) => {
            if let Some(warning) = check_sm_count(device_sms, atlas_kernels::TARGET_SM_COUNT) {
                tracing::warn!("{warning}");
            }
        }
        Err(e) => tracing::debug!("SM-count cross-check skipped: {e}"),
    }
    Ok(())
}

#[cfg(test)]
#[path = "arch_preflight_tests.rs"]
mod tests;
