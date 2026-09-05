// SPDX-License-Identifier: AGPL-3.0-only

//! cuda_backend unit tests. Pure CPU — no `cuInit`, no GPU touch — so
//! they run on every CI host.

use std::ffi::c_void;

use atlas_core::registry::RawCudaFunc;

use super::{effective_free_bytes, polled_free_bytes};
use crate::gpu::{DevicePtr, KernelHandle};

#[test]
fn kernel_handle_roundtrip() {
    // Verify KernelHandle <-> RawCudaFunc pointer conversion is lossless.
    let fake_ptr = 0xDEAD_BEEF_CAFE_u64;
    let handle = KernelHandle(fake_ptr);
    let raw = RawCudaFunc(handle.0 as *mut c_void);
    let back = raw.0 as u64;
    assert_eq!(back, fake_ptr);
}

#[test]
fn null_free_is_noop() {
    // AtlasCudaBackend::free should handle null pointers gracefully.
    // Can't call without GPU, but verify the DevicePtr::is_null logic.
    assert!(DevicePtr::NULL.is_null());
    assert!(!DevicePtr(0x1000).is_null());
}

// ── free-memory reporting rule ──────────────────────────────────────
//
// `effective_free_bytes` and `polled_free_bytes` are pure, so the rule that
// decides whether host `MemAvailable` may stand in for device free memory is
// tested here without a GPU or a CUDA context.

const GIB: usize = 1024 * 1024 * 1024;

/// The RTX PRO 6000 numbers this bug was found on: a 95 GiB card with 4.2 GiB
/// actually free, on a host reporting MemAvailable 1,038,438,936 kB (~990 GiB).
const DISCRETE_CU_FREE: usize = 4 * GIB + 280 * 1024 * 1024;
const HUGE_HOST_MEM_AVAILABLE: usize = 1_038_438_936 * 1024;

#[test]
fn discrete_device_ignores_host_mem_available() {
    // Host RAM is a different pool; substituting it made free_memory() report
    // ~990 GB and the KV pool was sized as if nothing had been allocated.
    assert_eq!(
        effective_free_bytes(DISCRETE_CU_FREE, Some(HUGE_HOST_MEM_AVAILABLE), false),
        DISCRETE_CU_FREE,
        "a discrete GPU must report the driver's device-free figure verbatim"
    );
}

#[test]
fn integrated_device_takes_the_max() {
    // GB10 (DGX Spark): device and host share one LPDDR5X pool, and
    // cuMemGetInfo reports MemFree, which excludes reclaimable buff/cache.
    assert_eq!(
        effective_free_bytes(20 * GIB, Some(90 * GIB), true),
        90 * GIB
    );
}

#[test]
fn integrated_device_without_meminfo_uses_driver_figure() {
    // /proc/meminfo missing or unparseable (non-Linux, container).
    assert_eq!(effective_free_bytes(20 * GIB, None, true), 20 * GIB);
}

#[test]
fn integrated_device_keeps_driver_figure_when_it_is_larger() {
    // MemAvailable can be the smaller of the two; `max` must not shrink the
    // driver's figure.
    assert_eq!(
        effective_free_bytes(60 * GIB, Some(10 * GIB), true),
        60 * GIB
    );
}

#[test]
fn watchdog_poll_on_a_discrete_device_reports_device_memory() {
    // What the OOM watchdog and the TUI gauge see. Before the integrated-only
    // rule reached this path, the poll returned ~990 GB on a 95 GB card, so
    // `free < threshold_bytes` was false forever and the watchdog could not
    // fire however close the device came to OOM.
    assert_eq!(
        polled_free_bytes(DISCRETE_CU_FREE, Some(HUGE_HOST_MEM_AVAILABLE), Some(false)),
        DISCRETE_CU_FREE
    );
}

#[test]
fn watchdog_poll_with_unknown_integration_is_treated_as_discrete() {
    // `None` = the driver would not say (cuCtxGetDevice or cuDeviceGetAttribute
    // failed). Guessing "integrated" would inflate the reading and disarm the
    // watchdog; guessing "discrete" only under-reports by the reclaimable
    // buff/cache on a machine where the max would have been right.
    assert_eq!(
        polled_free_bytes(DISCRETE_CU_FREE, Some(HUGE_HOST_MEM_AVAILABLE), None),
        DISCRETE_CU_FREE,
        "an unknown integrated/discrete answer must never inflate device free memory"
    );
}

#[test]
fn watchdog_poll_on_an_integrated_device_still_takes_the_max() {
    assert_eq!(
        polled_free_bytes(20 * GIB, Some(90 * GIB), Some(true)),
        90 * GIB
    );
}
