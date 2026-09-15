// SPDX-License-Identifier: AGPL-3.0-only

//! The low-rank collapse's three PROJECTION launches — split from
//! `hyper_connection_lowrank_gemm.rs` (500-LoC cap).
//!
//! `hc_pre_gemm_fused` picks between these per arm; neither is specific to
//! the mHC highway beyond its shapes.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// `ATLAS_HC_DENSE_GEMV=1` (presence) routes decode-shaped rows through the
/// batched dense GEMV instead of cuBLASLt. OPT-IN, default OFF — measured
/// 2026-09-05 on qwen3.8-flash-next EXL3 (GB10, 2 drafts, prefix cache on,
/// fresh server per arm): the GEMV arm was faster per kernel but draft
/// acceptance fell 1.47 → 1.37 per step and decode 30.36 → 29.05 tok/s;
/// serial 23.48 → 23.70 (noise). Serial, verify and the MTP draft module
/// all took the same row-invariant kernel, so this is not a serial/verify
/// mismatch — the dense GEMV's numerics themselves cost draft agreement.
/// Kept as an A/B arm; records in .research/exl3_decode_perf/ab_hc_*.
fn hc_dense_gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_HC_DENSE_GEMV").is_some())
}

/// Opt-in arm (see [`hc_dense_gemv_enabled`]): decode-shaped rows
/// (`m <= 8`) through `dense_gemv_bf16_batchm` — one pass over the `[n, k]`
/// BF16 weight for all rows, bit-identical per row to the M=1 kernel (fixed
/// K order, `--fmad=false`), one launch and no reduce kernel. Default: the
/// cuBLASLt path (a kernel per M with split-K + reduce; under the row-exact
/// contract one M=1 call per row).
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_rows(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    row_exact: bool,
    stream: u64,
) -> Result<()> {
    if m <= 8 && k.is_multiple_of(8) && n.is_multiple_of(4) && hc_dense_gemv_enabled() {
        let kernel = gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
        let dw = crate::weight_map::DenseWeight { weight: w };
        return crate::layers::ops::dense_gemv_batchm(gpu, kernel, a, &dw, out, m, n, k, n, stream);
    }
    let batch = if row_exact { 1 } else { m };
    for row in (0..m).step_by(batch as usize) {
        crate::layers::ops::cublas_bf16_proj_dense(
            a.offset(row as usize * k as usize * 2),
            w,
            out.offset(row as usize * n as usize * 2),
            batch,
            n,
            k,
            stream,
        )?;
    }
    Ok(())
}

pub(crate) fn gemm_raw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(128), m.div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
