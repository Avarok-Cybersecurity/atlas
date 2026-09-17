// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime BF16 → FP8 (E4M3) weight quantization with 128×128 block scales.
//!
//! The third runtime quantizer, and the one that exists for MoE experts.
//! Atlas already had:
//!   - BF16 → NVFP4 (`quantize_to_nvfp4`) — 4 bits, what `Bf16Raw` models get
//!   - BF16 → FP8 per-ROW (`quantize_bf16_to_fp8`) — for the DFlash head
//!
//! Neither fits routed experts on a plain-BF16 checkpoint. NVFP4 costs real
//! output quality, and the FP8 grouped MoE GEMM
//! (`moe_fp8_grouped_gemm.cu`) reads BLOCK scales `[N/128, K/128]` FP32,
//! not per-row ones. Feeding it a per-row buffer is not a shape error the
//! kernel can detect — it would index a shorter array and silently dequant
//! with the wrong scale, which is why `Fp8Weight::scale_format` is tagged
//! and asserted at dispatch.
//!
//! Why block-scaled FP8 rather than BF16 for experts: on LongCat-Flash-Lite
//! the routed experts are 63.0 GB of the 70.2 GB resident. BF16 does not fit
//! (+47 GB against a 97.3 GB budget at 0.80 util); FP8 does (+15.75 GB). And
//! because MoE reads only top-12 of 256 experts per token, the DECODE cost is
//! +0.74 GB/token — less than either of the dense BF16 levers.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;

use super::{DenseWeight, Fp8Weight, WeightQuantFormat};

/// Elements per block scale, both axes. Must match `FP8_BLOCK` in
/// `moe_fp8_grouped_gemm.cu` — the consumer hardcodes 128.
const FP8_BLOCK: usize = 128;

/// Quantize an `[n, k]` BF16 dense weight to block-scaled FP8 E4M3 on GPU.
///
/// Returns a `Fp8Weight` tagged `Fp8BlockScaled`, laid out exactly as the
/// on-disk Qwen FP8 releases are after widening, so every consumer that
/// already accepts those accepts this with no change.
///
/// Called once per projection at load time, never on the hot path. The BF16
/// source is the caller's to free — this does not take ownership.
pub fn quantize_to_fp8_blockscaled(
    bf16_weight: &DenseWeight,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    quantize_kernel: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<Fp8Weight> {
    anyhow::ensure!(
        n > 0 && k > 0,
        "quantize_to_fp8_blockscaled: empty [{n},{k}]"
    );

    let n_blocks = n.div_ceil(FP8_BLOCK);
    let k_blocks = k.div_ceil(FP8_BLOCK);

    // One byte per weight, one f32 per [128,128] tile.
    let weight_buf = gpu.alloc(n * k)?;
    let scale_buf = gpu.alloc(n_blocks * k_blocks * 4)?;

    KernelLaunch::new(gpu, quantize_kernel)
        .grid([k_blocks as u32, n_blocks as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16_weight.weight)
        .arg_ptr(weight_buf)
        .arg_ptr(scale_buf)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    Ok(Fp8Weight {
        weight: weight_buf,
        row_scale: scale_buf,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    })
}
