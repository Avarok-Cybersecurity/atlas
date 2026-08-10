// SPDX-License-Identifier: AGPL-3.0-only

//! GEMM-path dispatch helpers + roofline instrumentation. Extracted from the
//! `ops` module root during the ≤500-line split. Re-exported at
//! `crate::layers::ops::*` via `ops.rs`.

#![allow(unused_imports)]

use super::*;

// The nine GEMM-path flags that lived here as `OnceLock<bool>` statics are now
// `layers::ops::GemmDispatch`, resolved once when the model is built and
// carried on `ForwardContext`. A static outlived the model whose flags it
// encoded — swap to a model with different levers and the process kept serving
// the previous model's dispatch decisions, silently. It also hid the
// dependency: a function reading the environment through a static takes no
// argument that says so and gives the compiler nothing to check.

use spark_runtime::gpu::GpuBackend;

pub fn log_cutlass_nvfp4_route(gpu: &dyn GpuBackend, name: &str, m: u32, n: u32, k: u32) {
    // Routing telemetry, not a warning: the dedup key includes M, and
    // prefill produces a new M per token count, so at WARN this spammed the
    // production channel on every agentic request (and a polluted WARN
    // stream misdirects real investigations). Skip the dedup probe entirely
    // unless a subscriber would take the debug event — this runs per routed
    // GEMM call.
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    // De-duplicated on the BACKEND (`OpCache::first_shape`), not in a static:
    // the shapes a model dispatches are its own, and a process-wide set
    // suppresses the first route line for every shape a previous model
    // happened to use — the lines that say which kernel this model took.
    if gpu.op_cache().first_shape(name, m, n, k) {
        tracing::debug!("CUTLASS_NVFP4_ROUTE {name} M={m} N={n} K={k}");
    }
}

/// Roofline instrumentation: log each unique (kernel, M, N, K) GEMM shape once,
/// gated by `ATLAS_GEMM_SHAPE_LOG=1`. Used to cross-reference nsys per-call
/// durations → achieved TFLOPS/bandwidth vs GB10 peak.
pub fn log_gemm_shape(gpu: &dyn GpuBackend, name: &str, m: u32, n: u32, k: u32) {
    if std::env::var("ATLAS_GEMM_SHAPE_LOG").ok().as_deref() != Some("1") {
        return;
    }
    if gpu.op_cache().first_shape(name, m, n, k) {
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        tracing::warn!("GEMM_SHAPE {name} M={m} N={n} K={k} FLOP={flop:.3e}");
    }
}
