// SPDX-License-Identifier: AGPL-3.0-only

//! Optional kernel lookups: the handle if the kernel is there, `0` if not —
//! and, for a module only some targets compile, no lookup at all where it
//! was never built.

use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// Probe a kernel that only SOME targets compile — a Hopper-owned twin, a
/// tensor-core tier the gb10 tree does not carry — without issuing a lookup
/// on a target that never built the module. The boot audit records every
/// failed lookup as a dispatch site on a silent fallback path and refuses to
/// serve; a target that does not carry the source has no fallback to be
/// silent about, it has its only path. Stack 1089308's first campaign found
/// 17 such lookups on GB10 and could not boot. On a target that DOES carry
/// the module this is exactly [`try_kernel`], audit included.
#[track_caller]
pub fn try_target_kernel(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    if !gpu.has_module(module) {
        return KernelHandle(0);
    }
    try_kernel(gpu, module, func)
}

#[track_caller]
pub fn try_kernel(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    match gpu.kernel(module, func) {
        Ok(h) => h,
        Err(_) => {
            tracing::debug!("Optional kernel '{module}::{func}' not loaded");
            KernelHandle(0)
        }
    }
}

#[cfg(test)]
#[path = "kernel_probe_tests.rs"]
mod kernel_probe_tests;
