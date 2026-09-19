// SPDX-License-Identifier: AGPL-3.0-only

//! Read-side data plane for the dashboard: pure pollers over process-global
//! state (prometheus counters, scheduler snapshot, kernel audit, HF cache).
//! Nothing here touches the scheduler thread's locals.

pub mod catalogue;
pub mod kernels;
pub mod library;
pub mod metrics_poll;
pub mod thermal;

/// Free GPU memory in bytes, or `None` where no such query exists.
///
/// `spark_runtime::cuda_backend` is behind the `cuda` feature, so a bare call
/// breaks the metal/CPU builds — which is exactly how the dashboard shipped:
/// two call sites, no cfg, and the macOS CI job could not compile the crate.
/// One gated accessor means the next caller cannot repeat that.
pub fn gpu_free_bytes() -> Option<usize> {
    #[cfg(feature = "cuda")]
    {
        spark_runtime::cuda_backend::cuda_free_memory_bytes()
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}
