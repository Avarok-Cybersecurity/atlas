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
// into one forward. CODISPATCH is now a command-line flag on the same terms as
// VARLEN below -- the flag→lever conversion those integration notes tracked.
// Q12 stays env-only: it is the older spelling of the same path, kept for
// recipes that predate the rename, and nothing measures it.

/// The resolved CODISPATCH decision. ONE cell, FIVE readers: the three
/// scheduler sites in `spark-server` (the admission window, the chunk-0 defer,
/// and the shared-geometry guard) and the two batched-first-chunk sites in this
/// crate. A `OnceLock` so the decision cannot change mid-serve.
///
/// ★ WHY IT BECAME A FLAG, 2026-09-22. `bench.yaml env:` is NODE-WIDE, so the
/// env spelling arms every gate on a box at once. `concurrency-sweep` wants
/// this lever and `ttft-warm-gate` cannot tolerate it: three control records at
/// 0 agree to 0.045% (193.469 / 193.528 / 193.556 ms) while the one run at 1 is
/// 202.760 ms -- +4.78%, about 107x that spread, against a +3.0% limit. Only a
/// per-gate flag can give one gate the lever and deny it to the other.
static PREFILL_CODISPATCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Publish the command line's `--prefill-codispatch` decision. Returns the
/// value IN FORCE, which differs from `enabled` when something already resolved
/// the cell (then the command line did NOT take effect -- the caller warns,
/// mirroring `set_prefill_varlen_from_cli`). Absent flag ⇒ never called ⇒ the
/// documented `AVAROK_PREFILL_CODISPATCH` fallback stays reachable.
pub fn set_prefill_codispatch_from_cli(enabled: bool) -> bool {
    let _ = PREFILL_CODISPATCH.set(enabled);
    *PREFILL_CODISPATCH.get().expect("just set")
}

/// Cross-request co-dispatch of fresh prompts enabled? (`--prefill-codispatch`,
/// legacy `AVAROK_PREFILL_CODISPATCH=1`; default OFF).
///
/// SSOT for all five readers. They must agree: the scheduler defers chunk-0 on
/// this decision and the model layer then decides whether the batched path is
/// eligible, so a disagreement strands streams mid-admission.
pub fn prefill_codispatch_enabled() -> bool {
    *PREFILL_CODISPATCH.get_or_init(|| {
        bool_value_enabled(std::env::var("AVAROK_PREFILL_CODISPATCH").ok().as_deref())
    })
}

/// Whether chunk-zero streams may use the paged batched-prefill path.
///
/// ★ THE OR MUST SURVIVE THE PROMOTION. In THIS crate codispatch is one of TWO
/// ways to enable the same path, OR-ed with the older `AVAROK_Q12_BATCHED_FIRST_CHUNK`
/// spelling; in `spark-server` it is a standalone scheduling switch. A 1:1
/// replacement of the env read here would have silently dropped the Q12 alias
/// and turned a path off for recipes that still set it. The two meanings are
/// why this reads as an explicit OR rather than sharing the scheduler's call.
pub fn prefill_batched_first_chunk_enabled() -> bool {
    prefill_batched_first_chunk_from_parts(
        prefill_codispatch_enabled(),
        std::env::var("AVAROK_Q12_BATCHED_FIRST_CHUNK")
            .ok()
            .as_deref(),
    )
}

/// The pure OR, kept separable so the rule is testable without touching a
/// process-wide `OnceLock` or the environment. Note the ASYMMETRY that the
/// promotion introduced and that the signature now states: codispatch arrives
/// ALREADY RESOLVED (flag, else env), while Q12 is still a raw env spelling.
fn prefill_batched_first_chunk_from_parts(codispatch: bool, q12: Option<&str>) -> bool {
    codispatch || bool_value_enabled(q12)
}

/// The resolved VARLEN batched-prefill decision. One cell, three readers
/// (admission predicate, batched-attention chunk-0 guard, scheduler wave
/// planner) — a `OnceLock` so the decision cannot change mid-serve.
static PREFILL_VARLEN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Publish the command line's `--prefill-varlen-batch` decision. Returns the
/// value IN FORCE, which differs from `enabled` when something already
/// resolved the cell (then the command line did NOT take effect — the caller
/// warns, mirroring `gdn_flags::set_from_cli`). Absent flag ⇒ never called ⇒
/// the documented `AVAROK_PREFILL_VARLEN` fallback stays reachable.
pub fn set_prefill_varlen_from_cli(enabled: bool) -> bool {
    let _ = PREFILL_VARLEN.set(enabled);
    *PREFILL_VARLEN.get().expect("just set")
}

/// VARLEN (ragged) batched prefill enabled? (`--prefill-varlen-batch`,
/// legacy `AVAROK_PREFILL_VARLEN=1`; default OFF).
///
/// SSOT for the admission predicate (`check_kernel_batched_eligible`), the
/// batched-attention layer's chunk-0 guard, and the scheduler's prefill wave
/// planner. Those must agree: if admission accepts a batch the layer then
/// rejects, the bail happens mid-Phase-A with streams already mutated, and
/// the per-stream fallback re-runs setup on dirty state.
pub fn prefill_varlen_enabled() -> bool {
    *PREFILL_VARLEN
        .get_or_init(|| bool_value_enabled(std::env::var("AVAROK_PREFILL_VARLEN").ok().as_deref()))
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
/// gated by `AVAROK_GEMM_SHAPE_LOG=1`. Used to cross-reference nsys per-call
/// durations → achieved TFLOPS/bandwidth vs GB10 peak.
pub fn log_gemm_shape(gpu: &dyn GpuBackend, name: &str, m: u32, n: u32, k: u32) {
    if std::env::var("AVAROK_GEMM_SHAPE_LOG").ok().as_deref() != Some("1") {
        return;
    }
    if gpu.op_cache().first_shape(name, m, n, k) {
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        tracing::warn!("GEMM_SHAPE {name} M={m} N={n} K={k} FLOP={flop:.3e}");
    }
}

#[cfg(test)]
mod tests {
    use super::{bool_value_enabled, prefill_batched_first_chunk_from_parts};

    #[test]
    fn accepts_boolean_environment_spellings() {
        assert!(bool_value_enabled(Some("1")));
        assert!(bool_value_enabled(Some("true")));
        assert!(bool_value_enabled(Some("TRUE")));
        assert!(!bool_value_enabled(Some("0")));
        assert!(!bool_value_enabled(Some("false")));
        assert!(!bool_value_enabled(None));
    }

    /// ★ THE ASSERTION THAT CATCHES A 1:1 PROMOTION. When codispatch became a
    /// command-line flag it stopped being an env read HERE too -- and the
    /// tempting edit was to replace both env reads with the one resolved
    /// decision. That would have dropped the Q12 alias silently, turning the
    /// batched chunk-0 path OFF for every recipe that still sets the older
    /// spelling and nothing but a production regression to say so. The third
    /// case below is the one that fails if anyone does it.
    #[test]
    fn either_chunk_zero_spelling_enables_admission() {
        // codispatch alone (the flag, or its env fallback, already resolved)
        assert!(prefill_batched_first_chunk_from_parts(true, None));
        // Q12 alone -- the alias that must survive the promotion
        assert!(prefill_batched_first_chunk_from_parts(false, Some("true")));
        assert!(prefill_batched_first_chunk_from_parts(false, Some("1")));
        // neither
        assert!(!prefill_batched_first_chunk_from_parts(false, None));
        // explicit off on both spellings
        assert!(!prefill_batched_first_chunk_from_parts(false, Some("0")));
        assert!(!prefill_batched_first_chunk_from_parts(
            false,
            Some("false")
        ));
        // ★ codispatch OFF does not veto Q12: it is an OR, not a master switch.
        // An `&&` here would read as "codispatch gates everything", which is
        // what the scheduler means by the word and NOT what this crate does.
        assert!(prefill_batched_first_chunk_from_parts(false, Some("1")));
    }
}
