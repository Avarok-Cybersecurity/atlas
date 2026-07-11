// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 (E4M3) dense GEMM ops — split out of `gemm_dense.rs` to keep each file
//! under the 500-LoC cap. Sibling of `gemm_dense_int8.rs`; re-exported through
//! `ops::*`, so call sites keep using `ops::fp8_gemm_n128(..)` unchanged.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Pre-dequanted FP8 GEMM (prefill): C = A @ B_fp8.
///
/// A: [M, K] BF16, B_fp8: [N, K] FP8 E4M3 (pre-dequanted from NVFP4), C: [M, N] BF16.
/// Eliminates runtime NVFP4→FP8 dequant — only LOAD + FP8 MMA per K step.
///
/// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // DEFAULT-ON: route the GDN-projection prefill GEMM through the ldmatrix.x4
    // A+B kernel (fp8_fp8_gemm_ldmab). ncu-proven 2.1x over the scalar-load
    // fp8_gemm_t (10.7%->higher SM, MIO stall 74->44 cyc), cosine 1.000000 vs
    // fp8_fp8_gemm_t, and a confirmed same-box+cross-box e2e warm-TTFT win
    // (median -5.8%, p90 -10.3%, IoU 0.6264 unchanged). Quantizes the bf16
    // activation to e4m3 once into a persistent scratch, then launches the
    // ldmatrix GEMM. K must be a multiple of 32 (the ldmab K-tile). Opt-OUT with
    // ATLAS_FP8_LDMAB=0 (falls through to the scalar path below).
    //
    // CAPABILITY-GATED: `fp8_fp8_gemm_ldmab` is currently only compiled for the
    // qwen3.6-27b/nvfp4 target, but this fn is shared by every model that takes the
    // FP8 prefill path (35b-a3b, qwen3-next-80b, gemma-4, minimax, deepseek-v4, ...).
    // So the handles are probed with `.ok()`, NOT `.expect()`: a target without the
    // kernel transparently falls through to the validated scalar `fp8_gemm_t` below
    // instead of panicking at first prefill.
    if k.is_multiple_of(32) && std::env::var("ATLAS_FP8_LDMAB").as_deref() != Ok("0") {
        use std::sync::{Mutex, OnceLock};
        static QK: OnceLock<Option<KernelHandle>> = OnceLock::new();
        static LK: OnceLock<Option<KernelHandle>> = OnceLock::new();
        static SCRATCH: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);
        let qk = *QK.get_or_init(|| gpu.kernel("w4a16", "bf16_to_fp8").ok());
        let lk = *LK.get_or_init(|| gpu.kernel("w4a16", "fp8_fp8_gemm_ldmab").ok());
        if let (Some(qk), Some(lk)) = (qk, lk) {
            let need = (m as usize) * (k as usize); // e4m3 bytes
            let a8 = {
                let mut g = SCRATCH.lock().unwrap();
                if g.map(|(_, sz)| sz < need).unwrap_or(true) {
                    let p = gpu.alloc(need)?; // grow-only; old ptr leaked (rare, per-run)
                    *g = Some((p, need));
                }
                g.unwrap().0
            };
            bf16_to_fp8(gpu, qk, input, a8, m * k, stream)?;
            return KernelLaunch::new(gpu, lk)
                .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(a8)
                .arg_ptr(b_fp8)
                .arg_ptr(output)
                .arg_u32(m)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream);
        }
    }
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Pre-dequant NVFP4 → FP8 E4M3.  One-time conversion at model load.
///
/// Reads B_packed[N, K/2] + B_scale[N, K/GROUP_SIZE] + scale2 → B_fp8[N, K].
///
/// Grid: (ceil(N*K/2 / 256), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn predequant_nvfp4_to_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    b_packed: DevicePtr,
    b_scale: DevicePtr,
    scale2: f32,
    b_fp8: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let total = n * k / 2;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b_packed)
        .arg_ptr(b_scale)
        .arg_f32(scale2)
        .arg_ptr(b_fp8)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Convert BF16 activations to FP8 E4M3 for FP8×FP8 GEMM.
///
/// Grid: (ceil(total_elements/2 / 256), 1, 1)  Block: (256, 1, 1)
pub fn bf16_to_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    total_elements: u32,
    stream: u64,
) -> Result<()> {
    let threads_needed = total_elements / 2;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(threads_needed, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(total_elements)
        .launch(stream)
}
