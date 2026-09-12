// SPDX-License-Identifier: AGPL-3.0-only

//! The device preflight's decisions, on a host with no CUDA at all.
//!
//! Its own file rather than a `#[cfg(test)]` module inside
//! `arch_preflight.rs`: that file is at the repository's 500-LoC cap and the
//! SM-count cross-check added to it (#928) has its own cases. The split is by
//! QUESTION — the module states the rules, this file drives them against
//! injected drivers — so each file still reads as one argument.

use super::{
    DeviceQuery, DriverDeviceQuery, check_arch, check_sm_count, device_compute_capability_of,
    preflight_arch, preflight_device_arch, preflight_device_arch_with,
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
        self.calls
            .lock()
            .expect("fake driver lock")
            .push((op, thread::current().id(), ordinal));
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

    /// The cross-check is BEST EFFORT: a driver that will not answer costs one
    /// cross-check, never the boot. This fake refuses, so every ordering case
    /// below also proves the arch preflight still returns `Ok` when it does.
    fn sm_count(&self, ordinal: usize) -> Result<u32> {
        self.log("sm_count", ordinal);
        bail!("this fake driver does not answer the SM count");
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
        // The SM-count cross-check (#928), addressed by the SAME ordinal and
        // run AFTER the arch verdict. This fake refuses it, which is how the
        // `expect` above also pins that a driver with no answer costs one
        // cross-check and not the boot.
        ("sm_count", _, 3),
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

// ── the SM-count cross-check (#928) ──

/// AGREEMENT is silent. A boot line per device fact is a boot line nobody
/// reads; the warning exists to be rare.
#[test]
fn a_matching_sm_count_says_nothing() {
    assert_eq!(check_sm_count(132, 132), None);
    assert_eq!(check_sm_count(48, 48), None);
}

/// ★ THE DEFECT, pinned. `atlas_core::device::sm121::NUM_SMS = 48` — a
/// constant named after ONE part — reaching a build whose kernels were sized
/// for another is exactly what went unnoticed for a whole campaign, because
/// nothing on either side ever said a number out loud.
///
/// The warning therefore names BOTH numbers and the file the declaration came
/// from: a log line an operator can act on without reading the source.
#[test]
fn a_mismatched_sm_count_warns_naming_both_numbers() {
    let warning = check_sm_count(132, 48).expect("48 declared, 132 present must warn");
    assert!(warning.contains("132"), "{warning}");
    assert!(warning.contains("48"), "{warning}");
    assert!(warning.contains("sm_count"), "{warning}");
    // …and it must not read as a refusal. A wrong count costs a suboptimal
    // grid, never a wrong answer, so the serve continues.
    assert!(warning.contains("serving is unaffected"), "{warning}");
}

/// The cross-check is ONE-WAY in neither direction: a build sized for MORE
/// SMs than the device has is the same defect seen from the other side (a
/// grid with nothing to fill), and must warn too.
#[test]
fn the_cross_check_fires_in_both_directions() {
    assert!(check_sm_count(48, 132).is_some());
    assert!(check_sm_count(132, 48).is_some());
}
