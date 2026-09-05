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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DeviceQuery, DriverDeviceQuery, check_arch, device_compute_capability_of, preflight_arch,
        preflight_device_arch, preflight_device_arch_with,
    };
    use anyhow::{Result, bail};
    use atlas_core::target::KernelTarget;
    use atlas_kernels::{ModelBehavior, SamplingPresets, TargetPtxSet};
    use std::sync::Mutex;
    use std::thread::{self, ThreadId};

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

    /// A fake driver reproducing the ONE property of the real one that the
    /// review finding turns on: `cuda_host::host` is a `OnceLock`, so only the
    /// FIRST call makes a context current on its calling thread; every later
    /// call hands back an `Arc` clone and leaves thread-current state alone.
    struct FakeDriver {
        /// The thread a context was actually made current on, set by the first
        /// `init_host` only — the scheduler thread, in the reported swap.
        ctx_current_on: Mutex<Option<ThreadId>>,
        /// Every call this driver took, as `(operation, thread, ordinal)`.
        calls: Mutex<Vec<(&'static str, ThreadId, usize)>>,
        /// `true` spells the query `cuCtxGetDevice`: it reads the CALLING
        /// thread's current context. `false` spells it `cuDeviceGet`.
        reads_current_context: bool,
    }

    impl FakeDriver {
        fn new(reads_current_context: bool) -> Self {
            Self {
                ctx_current_on: Mutex::new(None),
                calls: Mutex::new(Vec::new()),
                reads_current_context,
            }
        }

        fn log(&self, op: &'static str, ordinal: usize) {
            self.calls.lock().expect("fake driver lock").push((
                op,
                thread::current().id(),
                ordinal,
            ));
        }
    }

    impl DeviceQuery for FakeDriver {
        fn init_host(&self, ordinal: usize) -> Result<()> {
            self.log("init_host", ordinal);
            let mut current = self.ctx_current_on.lock().expect("fake driver lock");
            if current.is_none() {
                *current = Some(thread::current().id());
            }
            Ok(())
        }

        fn compute_capability(&self, ordinal: usize) -> Result<(u32, u32)> {
            self.log("compute_capability", ordinal);
            if self.reads_current_context
                && *self.ctx_current_on.lock().expect("fake driver lock")
                    != Some(thread::current().id())
            {
                // CUDA_ERROR_INVALID_CONTEXT.
                bail!("cuCtxGetDevice failed: status 201");
            }
            Ok((9, 0))
        }
    }

    /// Run `preflight_device_arch_with` on a thread that is NOT this one.
    fn preflight_on_a_fresh_thread(driver: &FakeDriver, ordinal: usize) -> Result<()> {
        thread::scope(|scope| {
            scope
                .spawn(|| preflight_device_arch_with(ordinal, Some("sm_90a"), driver))
                .join()
                .expect("the preflight thread must not panic")
        })
    }

    /// ★ THE DEFECT, pinned. A context-addressed capability query cannot run
    /// on a thread that did not create the process CUDA host.
    ///
    /// Oracle: the reported call chain. `cuda_host::host` binds only while
    /// initialising its `OnceLock`; a TUI Library swap with no adapters starts
    /// a fresh `atlas-swap` thread and tears the old model down on the
    /// scheduler thread, so nothing binds a context on the swap thread. NVIDIA
    /// documents the thread-current requirement in the context API, and the
    /// driver's answer is `CUDA_ERROR_INVALID_CONTEXT` (201) — which fails the
    /// requested load AND its restoration, leaving the host with no model.
    ///
    /// This is a static call-chain finding, not a GPU reproduction, so the
    /// driver is faked; what is asserted is the ordering contract, which is
    /// the part that was wrong.
    #[test]
    fn a_context_addressed_query_fails_on_a_thread_that_did_not_make_the_host() {
        let driver = FakeDriver::new(true);
        // Thread A: the scheduler thread that loaded the previous model.
        driver.init_host(3).expect("thread A creates the host");
        // Thread B: the fresh `atlas-swap` thread.
        let err = preflight_on_a_fresh_thread(&driver, 3)
            .expect_err("no context is current on the swap thread");
        assert!(
            format!("{err}").contains("201"),
            "expected CUDA_ERROR_INVALID_CONTEXT, got: {err}"
        );
    }

    /// …and the shipped shape, which is context-free, does not care.
    ///
    /// Oracle: `cuDeviceGet` resolves a `CUdevice` from an ORDINAL and reads
    /// no thread-current state, so `preflight_device_arch` — which is the
    /// ordinal it was handed, all the way down — is correct on any thread.
    #[test]
    fn the_ordinal_addressed_query_preflights_from_any_thread() {
        let driver = FakeDriver::new(false);
        driver.init_host(3).expect("thread A creates the host");
        preflight_on_a_fresh_thread(&driver, 3)
            .expect("an ordinal-addressed query needs no context of its own");

        let calls = driver.calls.lock().expect("fake driver lock");
        // Thread A's manual init, then the swap thread's whole sequence: the
        // host is still initialised first (for `cuInit`, and so the backend
        // reuses this context), and BOTH halves are addressed by the ordinal
        // the caller asked for rather than by whatever device some thread's
        // context happens to point at. That is the fix.
        let [
            (_, thread_a, 3),
            ("init_host", thread_b, 3),
            ("compute_capability", queried_on, 3),
        ] = calls[..]
        else {
            panic!("unexpected driver call sequence: {calls:?}");
        };
        assert_eq!(thread_b, queried_on, "both ran on the swap thread");
        // …and that thread is not the one the context was made current on,
        // which is the situation the old spelling could not survive.
        assert_ne!(
            thread_a, thread_b,
            "the defect only bites when these differ"
        );
    }

    /// The real driver, on a thread that did not create the host — the
    /// reported chain, unfaked. Needs a CUDA device, so it is `#[ignore]`d
    /// like the other GPU tests in this crate.
    #[test]
    #[ignore = "requires a free CUDA device"]
    fn the_real_preflight_runs_on_a_thread_that_did_not_make_the_host() {
        DriverDeviceQuery
            .init_host(0)
            .expect("this thread creates the process CUDA host");
        thread::spawn(|| {
            let (major, minor) =
                device_compute_capability_of(0).expect("cuDeviceGet needs no current context");
            assert!(major > 0, "driver reported CC {major}.{minor}");
            // Judge the arch the device itself reports, so this asserts the
            // call chain rather than which card the runner happens to hold.
            let arch = format!("sm_{major}{minor}");
            preflight_device_arch(0, Some(arch.as_str()))
        })
        .join()
        .expect("the preflight thread must not panic")
        .expect("a device's own compute capability must pass its preflight");
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
