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

// The two BATCHED-PREFILL ADMISSION flags below are not GEMM-path dispatch and
// have no `GemmDispatch` field; they gate whether concurrent prefills co-admit
// into one forward. They stay env reads for now (flag→lever conversion is
// per-PR follow-up work, tracked in the integration notes).

/// Whether chunk-zero streams may use the paged batched-prefill path.
///
/// `ATLAS_PREFILL_CODISPATCH` is the end-to-end request-admission flag;
/// keep the older Q12 spelling as a compatibility alias for existing recipes.
pub fn prefill_batched_first_chunk_enabled() -> bool {
    prefill_batched_first_chunk_from_values([
        std::env::var("ATLAS_Q12_BATCHED_FIRST_CHUNK")
            .ok()
            .as_deref(),
        std::env::var("ATLAS_PREFILL_CODISPATCH").ok().as_deref(),
    ])
}

fn prefill_batched_first_chunk_from_values(values: [Option<&str>; 2]) -> bool {
    values.into_iter().any(bool_value_enabled)
}

/// The resolved VARLEN batched-prefill decision. One cell, three readers
/// (admission predicate, batched-attention chunk-0 guard, scheduler wave
/// planner) — a `OnceLock` so the decision cannot change mid-serve.
static PREFILL_VARLEN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Publish the command line's `--prefill-varlen-batch` decision. Returns the
/// value IN FORCE, which differs from `enabled` when something already
/// resolved the cell (then the command line did NOT take effect — the caller
/// warns, mirroring `gdn_flags::set_from_cli`). Absent flag ⇒ never called ⇒
/// the documented `ATLAS_PREFILL_VARLEN` fallback stays reachable.
pub fn set_prefill_varlen_from_cli(enabled: bool) -> bool {
    let _ = PREFILL_VARLEN.set(enabled);
    *PREFILL_VARLEN.get().expect("just set")
}

/// VARLEN (ragged) batched prefill enabled? (`--prefill-varlen-batch`,
/// legacy `ATLAS_PREFILL_VARLEN=1`; default OFF).
///
/// SSOT for the admission predicate (`check_kernel_batched_eligible`), the
/// batched-attention layer's chunk-0 guard, and the scheduler's prefill wave
/// planner. Those must agree: if admission accepts a batch the layer then
/// rejects, the bail happens mid-Phase-A with streams already mutated, and
/// the per-stream fallback re-runs setup on dirty state.
pub fn prefill_varlen_enabled() -> bool {
    *PREFILL_VARLEN
        .get_or_init(|| bool_value_enabled(std::env::var("ATLAS_PREFILL_VARLEN").ok().as_deref()))
}

fn bool_value_enabled(value: Option<&str>) -> bool {
    matches!(value, Some("1")) || value.is_some_and(|value| value.eq_ignore_ascii_case("true"))
}

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

#[cfg(test)]
mod tests {
    use super::{bool_value_enabled, prefill_batched_first_chunk_from_values};

    #[test]
    fn accepts_boolean_environment_spellings() {
        assert!(bool_value_enabled(Some("1")));
        assert!(bool_value_enabled(Some("true")));
        assert!(bool_value_enabled(Some("TRUE")));
        assert!(!bool_value_enabled(Some("0")));
        assert!(!bool_value_enabled(Some("false")));
        assert!(!bool_value_enabled(None));
    }

    #[test]
    fn either_chunk_zero_spelling_enables_admission() {
        assert!(prefill_batched_first_chunk_from_values([Some("1"), None]));
        assert!(prefill_batched_first_chunk_from_values([
            None,
            Some("true")
        ]));
        assert!(!prefill_batched_first_chunk_from_values([None, None]));
        assert!(!prefill_batched_first_chunk_from_values([
            Some("0"),
            Some("false")
        ]));
    }
}
