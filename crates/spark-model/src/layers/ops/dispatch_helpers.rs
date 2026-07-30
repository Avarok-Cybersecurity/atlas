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

pub fn log_cutlass_nvfp4_route(name: &str, m: u32, n: u32, k: u32) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    // STATIC, DELIBERATELY. This is log de-duplication and nothing else: it
    // records which (projection, shape) combinations have already been printed
    // so the route line appears once instead of once per token. It holds no
    // model-derived value — the tuple is a name hash and three dimensions, all
    // of which are re-derived from the arguments on every call — so a stale
    // entry cannot produce a wrong answer, only a suppressed duplicate log
    // line. Carrying it on ForwardContext would thread a logging concern
    // through every dispatch signature to prevent a repeated INFO line after a
    // model swap. The one real cost is that the first route line for a shape
    // the previous model also used is suppressed; `advance()`-scoping it would
    // be more code than the problem is worth.
    static SEEN: OnceLock<Mutex<HashSet<(u64, u32, u32, u32)>>> = OnceLock::new();
    let mut h: u64 = 1469598103934665603;
    for b in name.bytes() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen.lock().unwrap().insert((h, m, n, k)) {
        tracing::warn!("CUTLASS_NVFP4_ROUTE {name} M={m} N={n} K={k}");
    }
}

/// Roofline instrumentation: log each unique (kernel, M, N, K) GEMM shape once,
/// gated by `ATLAS_GEMM_SHAPE_LOG=1`. Used to cross-reference nsys per-call
/// durations → achieved TFLOPS/bandwidth vs GB10 peak.
pub fn log_gemm_shape(name: &str, m: u32, n: u32, k: u32) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    if std::env::var("ATLAS_GEMM_SHAPE_LOG").ok().as_deref() != Some("1") {
        return;
    }
    static SEEN: OnceLock<Mutex<HashSet<(u64, u32, u32, u32)>>> = OnceLock::new();
    let mut h: u64 = 1469598103934665603;
    for b in name.bytes() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    let key = (h, m, n, k);
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen.lock().unwrap().insert(key) {
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        tracing::warn!("GEMM_SHAPE {name} M={m} N={n} K={k} FLOP={flop:.3e}");
    }
}
