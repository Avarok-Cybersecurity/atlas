// SPDX-License-Identifier: AGPL-3.0-only

//! GEMM-path dispatch helpers + roofline instrumentation. Extracted from the
//! `ops` module root during the ≤500-line split. Re-exported at
//! `crate::layers::ops::*` via `ops.rs`.

#![allow(unused_imports)]

use super::*;

/// Whether block-scaled FP8 prefill (per-128-block weight scales + per-token
/// activation scales via `fp8_gemm_t_blockscaled` / `moe_w8a8_grouped_gemm`)
/// is enabled. This is the DEFAULT for block-scaled FP8 checkpoints as of
/// 2026-06-17: it matches vLLM's per-block precision and avoids the
/// single-scale `fp8_gemm_n128` path, whose collapse of per-block dynamic
/// range pushed long-context tool-arg decode into the FP8 argmax-flip regime
/// (B1 drift gauge ~1400 → ~100 once block-scaled prefill is on).
///
/// Opt out with `ATLAS_FP8_SINGLE_SCALE=1` to restore the old single-scale
/// prefill (diagnostic / fallback only). Call sites still guard on the
/// presence of block-scaled weights + kernel handles, so builds/models
/// without those fall back automatically regardless of this flag.
pub fn fp8_blockscaled_prefill_enabled() -> bool {
    !matches!(
        std::env::var("ATLAS_FP8_SINGLE_SCALE").ok().as_deref(),
        Some("1")
    )
}

/// cuBLASLt GEMM path enabled? (`ATLAS_CUBLAS_GEMM=1`), cached. The hand-written
/// mma.sync projection GEMMs hit only ~30% of the cuBLAS bf16 ceiling on GB10.
pub fn cublas_gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_CUBLAS_GEMM").ok().as_deref() == Some("1"))
}

/// Native-FP8 cuBLASLt GEMM path enabled? (`ATLAS_CUBLAS_FP8=1`), cached.
pub fn cublas_fp8_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_CUBLAS_FP8").ok().as_deref() == Some("1"))
}

/// CUTLASS GEMM path enabled? (`ATLAS_CUTLASS_GEMM=1`), cached. M0 is scoped to
/// dense BF16 projections using the same FP8→BF16 cached dequant as cuBLASLt.
pub fn cutlass_gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_CUTLASS_GEMM").ok().as_deref() == Some("1"))
}

/// Native CUTLASS NVFP4 GEMM path enabled? (`ATLAS_CUTLASS_NVFP4_GEMM=1`).
/// This path quantizes activations to CUTLASS NVFP4 and consumes transposed
/// Atlas NVFP4 weights after repacking scales into CUTLASS SM120 layout.
pub fn cutlass_nvfp4_gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var("ATLAS_CUTLASS_NVFP4_GEMM").ok().as_deref() == Some("1"))
}

fn cutlass_nvfp4_flag_enabled(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

/// Native CUTLASS NVFP4 SSM QKVZ path enabled.
pub fn cutlass_nvfp4_qkvz_enabled() -> bool {
    cutlass_nvfp4_gemm_enabled() || cutlass_nvfp4_flag_enabled("ATLAS_CUTLASS_NVFP4_QKVZ")
}

/// Native CUTLASS NVFP4 attention Q/K/V path enabled for the named projection.
pub fn cutlass_nvfp4_attn_qkv_enabled(label: &str) -> bool {
    cutlass_nvfp4_gemm_enabled()
        || match label {
            "q_proj" => cutlass_nvfp4_flag_enabled("ATLAS_CUTLASS_NVFP4_ATTN_Q"),
            "k_proj" | "v_proj" => cutlass_nvfp4_flag_enabled("ATLAS_CUTLASS_NVFP4_ATTN_KV"),
            _ => false,
        }
}

/// Native CUTLASS NVFP4 attention O path enabled.
pub fn cutlass_nvfp4_attn_o_enabled() -> bool {
    cutlass_nvfp4_gemm_enabled() || cutlass_nvfp4_flag_enabled("ATLAS_CUTLASS_NVFP4_ATTN_O")
}

/// Native CUTLASS NVFP4 SSM out-projection path enabled.
pub fn cutlass_nvfp4_ssm_out_enabled() -> bool {
    cutlass_nvfp4_flag_enabled("ATLAS_CUTLASS_NVFP4_SSM_OUT")
}

pub fn log_cutlass_nvfp4_route(name: &str, m: u32, n: u32, k: u32) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
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
