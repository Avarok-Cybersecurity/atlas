// SPDX-License-Identifier: AGPL-3.0-only

//! Row-wise FP8 projection routing.
//!
//! Split from `dispatch_proj.rs` when the cluster this branch added took that
//! file over the 500-LoC cap. It is a cohesive unit rather than an arbitrary
//! cut: the passthrough decision, the cached block->row-wise requant it guards,
//! and the router that consumes both.
//!
//! ⚠ The GEMM at the end of this path — `fp8_gemm_act_weight_t_rowwise` —
//! returns NOT_SUPPORTED on sm_121 (measured 2026-08-15, reproduced through the
//! block-scaled path with `ATLAS_CUBLAS_FP8=1`, so it is the GEMM and not the
//! weights). The mixed-precision loader therefore routes through
//! `cublas_bf16_proj` instead; see `weight_loader/qwen35_dense/rowwise_fp8.rs`.
//! This module stays because the passthrough is what a working per-row FP8
//! kernel would plug into.

// Everything here names its paths explicitly (`super::DerivedWeights`,
// `crate::weight_map::…`), so this file needs no glob import — unlike
// `dispatch_proj.rs`, which carries `#![allow(unused_imports)]` and a `use
// super::*`.

/// `(weight, scale)` verbatim when `fp8w` is ALREADY the row-wise pair the
/// cuBLASLt row-wise GEMM wants, else `None`.
///
/// Pure, and split out from the GPU path so the invariant is testable on a
/// CPU-only runner: the whole claim is "a row-wise checkpoint is passed
/// through untouched", and that is a decision about a tag, not about a device.
pub(super) fn rowwise_pair_passthrough(fp8w: &crate::weight_map::Fp8Weight) -> Option<(u64, u64)> {
    use crate::weight_map::WeightQuantFormat;
    (fp8w.scale_format == WeightQuantFormat::Fp8PerRow).then_some((fp8w.weight.0, fp8w.row_scale.0))
}

/// Re-quantize a block-scaled FP8 weight `[N,K]` → ROW-WISE FP8 (E4M3 + per-row
/// FP32 scale `[N]`) on-GPU once, cached by the FP8 weight pointer. Path:
/// block-fp8 → BF16 (transient) → row-wise fp8. Backs the GB10-supported
/// `cublas_fp8_rowwise_proj`. Returns `(fp8_weight_ptr, per_row_scale_ptr)`.
///
/// A weight that is ALREADY row-wise returns its own pointers untouched — see
/// the early return, which is the whole point of the `scale_format` tag here.
fn requant_weight_rowwise_fp8_cached(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    derived: &super::DerivedWeights,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<(u64, u64)> {
    use crate::weight_map::WeightQuantFormat;
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

    // ── Already row-wise: nothing to do, and nothing to lose ──────────────
    //
    // A mixed-precision compressed-tensors checkpoint (e.g.
    // unsloth/Qwen3.8-27B-NVFP4, `format = mixed-precision`) ships its
    // attention and GDN projections as FP8 E4M3 with a PER-CHANNEL scale —
    // which is exactly `(weight, [N] f32)`, the pair this function exists to
    // produce. Converting it would mean fp8 → bf16 → fp8, losing precision to
    // manufacture something it already is.
    //
    // Without this arm those checkpoints take the loader's fallback instead:
    // dequant to BF16 and RE-quantise to NVFP4, i.e. 8-bit weights served at
    // 4 bits. Measured on the video benchmark's hardest leg, that fallback
    // answered "Red, Blue" where the natively-loaded FP8 build of the same
    // weights managed "Red, Blue, Yellow".
    if let Some(pair) = rowwise_pair_passthrough(fp8w) {
        return Ok(pair);
    }
    // The conversion below reads `row_scale` as a `[N/128, K/128]` FP32 grid.
    // Anything else here is a caller bug, and a silent one — the buffer is
    // smaller than the grid, so it reads in-bounds garbage rather than
    // faulting. Assert instead.
    fp8w.scale_format
        .expect(WeightQuantFormat::Fp8BlockScaled, "rowwise-fp8 requant");
    let cache_key = fp8w.weight.0;
    if let Some(hit) = derived.get_pair(super::Derivation::RowwiseFp8, cache_key) {
        return Ok(hit);
    }
    let (n, k) = (fp8w.n, fp8w.k);
    // 1. block-fp8 → BF16 (transient scratch, freed after re-quant).
    let bf16 = gpu.alloc(n as usize * k as usize * 2)?;
    let block = 128u32;
    let sk = k / block;
    let dq = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, dq)
        .grid([div_ceil(k, 64), div_ceil(n, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(fp8w.weight)
        .arg_ptr(fp8w.row_scale)
        .arg_ptr(bf16)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(block)
        .arg_u32(block)
        .arg_u32(sk)
        .arg_u32(1)
        .launch(stream)?;
    // 2. BF16 → row-wise fp8 [N,K] + per-row scale [N].
    let w_fp8 = gpu.alloc(n as usize * k as usize)?;
    let w_scale = gpu.alloc(n as usize * 4)?;
    let qk = gpu.kernel("quant_rowwise_fp8", "quant_rowwise_fp8")?;
    KernelLaunch::new(gpu, qk)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16)
        .arg_ptr(w_fp8)
        .arg_ptr(w_scale)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    gpu.synchronize(stream)?; // re-quant must finish before the transient bf16 is freed
    gpu.free(bf16)?;
    derived.insert_pair(
        super::Derivation::RowwiseFp8,
        cache_key,
        (w_fp8.0, w_scale.0),
    );
    Ok((w_fp8.0, w_scale.0))
}

/// Route a projection through ROW-WISE native-FP8 cuBLASLt (the fp8 path GB10
/// supports). Weight is re-quantized once to per-row fp8 (cached); the activation
/// is quantized per-token each call. ~1.8× the bf16 path (152 vs 85 TF), and
/// frees the bf16-dequant memory the bf16 path holds.
/// `act_fp8_scratch` ≥ m*k fp8 bytes; `act_scale_scratch` ≥ m f32 (e.g. the
/// `buffers.fp8_act` / `fp8_act_scale` arena buffers).
#[allow(clippy::too_many_arguments)]
pub fn cublas_fp8_rowwise_proj(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    derived: &super::DerivedWeights,
    act_bf16: spark_runtime::gpu::DevicePtr,
    act_fp8_scratch: spark_runtime::gpu::DevicePtr,
    act_scale_scratch: spark_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: spark_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    use spark_runtime::kernel_args::KernelLaunch;
    let (w_fp8, w_scale) = requant_weight_rowwise_fp8_cached(gpu, derived, fp8w, stream)?;
    // Per-token row-wise quant of the activation → fp8 [M,K] + scale [M].
    let qk = gpu.kernel("quant_rowwise_fp8", "quant_rowwise_fp8")?;
    KernelLaunch::new(gpu, qk)
        .grid([m, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(act_bf16)
        .arg_ptr(act_fp8_scratch)
        .arg_ptr(act_scale_scratch)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)?;
    // ── M must be padded, exactly as the block-scaled sibling pads it ────
    //
    // Both scale vectors are declared `SCALE_MODE_OUTER_VEC_32F`, and
    // cuBLASLt will not serve an outer-vector extent that is not a multiple
    // of 4; `AlgoGetHeuristic` returns status 15 (NOT_SUPPORTED) rather than
    // failing at launch. Unpadded, this path worked only for callers whose M
    // happened to be aligned — a 23-token prompt through the row-wise GDN
    // prefill arm is what surfaced it, since a chunk size is whatever the
    // prompt is.
    //
    // Pad to 16 like `cublas_fp8_proj` (TC-friendly), and zero BOTH the
    // padding scales and the padding activation rows: with a zero scale the
    // phantom rows contribute nothing, and zeroed bytes cannot carry a NaN
    // into an accumulator. The phantom output rows are ignored by the
    // caller, same contract as the block-scaled path.
    let m_pad = m.div_ceil(16) * 16;
    if m_pad > m {
        let pad_rows = (m_pad - m) as usize;
        gpu.memset_async(
            act_scale_scratch.offset(m as usize * 4),
            0,
            pad_rows * 4,
            stream,
        )?;
        gpu.memset_async(
            act_fp8_scratch.offset(m as usize * k as usize),
            0,
            pad_rows * k as usize,
            stream,
        )?;
    }
    spark_runtime::cublaslt::fp8_gemm_act_weight_t_rowwise(
        act_fp8_scratch.0,
        act_scale_scratch.0,
        w_fp8,
        w_scale,
        out.0,
        m_pad,
        n,
        k,
        stream,
    )
}

#[cfg(test)]
mod rowwise_passthrough_tests {
    use super::{requant_weight_rowwise_fp8_cached, rowwise_pair_passthrough};
    use crate::layers::ops::DerivedWeights;
    use crate::weight_map::{Fp8Weight, WeightQuantFormat};
    use spark_runtime::gpu::DevicePtr;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn weight(scale_format: WeightQuantFormat) -> Fp8Weight {
        Fp8Weight {
            weight: DevicePtr(0xBEEF),
            row_scale: DevicePtr(0x5CA1E),
            n: 4096,
            k: 5120,
            scale_format,
        }
    }

    /// ★ The point of the change: a checkpoint that already ships per-row
    /// scales is handed to the row-wise GEMM untouched. Converting it would be
    /// fp8 -> bf16 -> fp8, spending precision to produce what it already is.
    #[test]
    fn an_already_rowwise_weight_passes_through_verbatim() {
        let w = weight(WeightQuantFormat::Fp8PerRow);
        assert_eq!(
            rowwise_pair_passthrough(&w),
            Some((w.weight.0, w.row_scale.0)),
            "the checkpoint's own pointers, not a converted copy"
        );
    }

    /// Every other format still takes the requant path — in particular
    /// block-scaled, which is what every current caller carries.
    #[test]
    fn other_formats_still_requantize() {
        for f in [
            WeightQuantFormat::Fp8BlockScaled,
            WeightQuantFormat::Fp8SingleScale,
            WeightQuantFormat::Bf16,
            WeightQuantFormat::Nvfp4,
        ] {
            assert_eq!(
                rowwise_pair_passthrough(&weight(f)),
                None,
                "{f:?} is not a row-wise pair and must not be passed through"
            );
        }
    }

    #[test]
    fn cached_requant_returns_rowwise_checkpoint_pointers_without_gpu_work() {
        let gpu = MockGpuBackend::new();
        let w = weight(WeightQuantFormat::Fp8PerRow);

        assert_eq!(
            requant_weight_rowwise_fp8_cached(&gpu, &DerivedWeights::new(), &w, 0).unwrap(),
            (w.weight.0, w.row_scale.0)
        );
        assert_eq!(gpu.alloc_count(), 0, "passthrough must not allocate a copy");
    }
}
