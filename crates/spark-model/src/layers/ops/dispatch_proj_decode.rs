// SPDX-License-Identifier: AGPL-3.0-only

//! W8A8 block-scaled cuBLASLt routing for the **5..16-row DECODE**
//! projections — the SSM `in_proj_qkvz`/`out_proj` and the attention
//! Q/K/V/O — plus the strided-output arithmetic those need and the dense FFN
//! did not.
//!
//! WHY (#927). H100, 2026-09-11 round 7, `Qwen/Qwen3.8-27B-FP8`, batch 16,
//! steady-state n=16 decode step **43.595 ms** (idle 4.2%), nsys with
//! `--cuda-graph-trace=node`. The dense FFN at these SAME 16 rows already runs
//! cuBLASLt W8A8 (`nvjet_sm90_…_Ablk128_Bvec128`, 128 launches = 5.8 ms +
//! split-K 2.4 ms + reduce 0.14 ms for 64 layers × 2 GEMMs ≈ **128 µs/layer**
//! for 267 MB of weights, i.e. ~2 100 GB/s-equivalent). The projections did
//! not, and they are now the largest line in the step:
//!
//! | kernel | launches | µs/step | what |
//! |---|---|---|---|
//! | `w8a16_gemv_batch16` GrdX 4096 | 48 | 11 294 | SSM `in_proj_qkvz` N=16384 K=5120, 235 µs each @ **357 GB/s** |
//! | `w8a16_gemv_batch16` GrdX 1280 | 64 | 6 814 | SSM `out_proj` + attn `o_proj`, N=5120, 106 µs each |
//! | `w8a16_gemv_batch16_strided` GrdX 3072 | 16 | 2 893 | attn `q_proj` N=12288, 181 µs each |
//! | `w8a16_gemv_batch16_strided` GrdX 256 | 32 | 846 | attn `k_proj`+`v_proj` N=1024, 26 µs each |
//!
//! 21.85 ms = **50.1% of the step** in one family of kernels at 357 GB/s
//! against a 3 350 GB/s HBM3 roofline, while the neighbouring FFN reaches
//! ~2 100 GB/s-equivalent on the same box in the same step. The GEMV reads the
//! weight once and then pays ~16 scalar FFMA per weight byte; the cuBLASLt
//! W8A8 path pays one `mma.sync.m16n8k32.e4m3` lane-slot instead. That is the
//! whole change: same weights, same block scales, vLLM's dynamic per-token
//! activation quant, a tensor-core MMA in place of the FFMA loop.
//!
//! NUMERICS. These rows move from W8A16 (BF16 activation × E4M3 weight) to
//! W8A8 (E4M3 activation, per-token 1×128 FP32 scales × the checkpoint's
//! 128×128 FP32 weight scales, FP32 epilogue) — **exactly the arithmetic the
//! dense FFN already uses at these same widths**, and vLLM's. It is a
//! deliberate precision trade, not a defect: E4M3 keeps 3 stored mantissa bits,
//! so the floor of a W8A8-vs-W8A16 comparison on these shapes is ~2-2.6%
//! relative RMS with cosine ~0.9997. The **M=1 path is untouched** (the
//! bit-exact scalar `w8a16_gemv`), and the batch oracles still compare the
//! GEMV tiers to the scalar loop bit-for-bit.
//!
//! GRAPH CAPTURE. Every selector here is a pure function of the PADDED ctx `n`
//! (the `padded_batch_n` ladder the graph cache is keyed by), the model's
//! resolved [`super::GemmDispatch`], and handles/capacities fixed at model
//! build. Nothing reads the environment per step.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{
    Fp8ActQuant, cublas_fp8_m_pad, cublas_scale_layout_kmajor, fp8_act_scale_to_kmajor,
    per_token_group_quant_fp8,
};

/// `ATLAS_NO_W8A8_DECODE_PROJ` kill switch: PRESENCE (any value, including
/// empty) keeps every 5..16-row decode projection on today's `w8a16_gemv_batch16`
/// tiers. Presence rather than `=1` for the same reason `ATLAS_FFN_W8A16_ONLY`
/// is presence-checked — an operator reaches for it while a serve is
/// misbehaving, and `...=0` meaning "on" is a trap.
///
/// `OnceLock`-cached and therefore constant for the life of the process, which
/// is what makes it safe to branch on under CUDA-graph capture.
pub fn w8a8_decode_proj_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("ATLAS_NO_W8A8_DECODE_PROJ").is_some())
}

/// The row band this family owns: 5..=16 PADDED decode rows.
///
/// * `<= 4` is `w8a16_gemv_batch4`'s, and M=1 is the bit-exact scalar GEMV —
///   at those widths the GEMV streams each weight once and no MMA tile beats
///   it, which the round-7 receipt measures directly (ncol2 at M=2/M=4 is 16%
///   and 11% SLOWER than the GEMV it replaces).
/// * `>= 17` is not a decode step on this ladder's hot rungs and would need a
///   different M pad; it stays on whatever tier owns it today.
pub const DECODE_W8A8_ROWS: std::ops::RangeInclusive<usize> = 5..=16;

/// Last element written by a cuBLASLt GEMM whose output rows are `ldc`
/// elements apart — SSOT for every strided call site's bounds check.
///
/// The D operand is a column-major `[N, M]` with leading dimension `ldc`,
/// i.e. a row-major `[M, N]` at row pitch `ldc`. The library writes `n`
/// elements of each of the `m_pad` columns, so the extent is
/// `(m_pad - 1) * ldc + n` ELEMENTS — NOT `m_pad * ldc`, which would count a
/// gap the GEMM never touches, and NOT `m_pad * n`, which would ignore the
/// pitch entirely.
///
/// ⚠ `m_pad`, not `m`: `cublas_fp8_proj_prequant` hands cuBLASLt `ceil16(M)`
/// and the phantom rows ARE written. With a contiguous output they land past
/// the live rows in the same buffer; with a STRIDED one they land in decode
/// slots `m..m_pad`, which belong to sequences that are not in this step. That
/// is in-bounds and harmless — those slots are re-projected before anything
/// reads them — but only while the buffer actually HAS `m_pad` slots, which is
/// exactly what this extent is used to check.
pub fn strided_out_extent_elems(m_pad: u32, ldc: u32, n: u32) -> usize {
    debug_assert!(m_pad >= 1);
    (m_pad as usize - 1) * ldc as usize + n as usize
}

/// Everything the decode W8A8 cuBLASLt arm needs, as a PURE function of shape,
/// format, lever and handles — so the CPU tests pin every clause without a GPU
/// and without touching the process environment.
///
/// Clauses, each load-bearing:
///
/// * `family_armed` — the caller's slice of [`super::CublasScope`]
///   (`cublas.ssm` / `cublas.attn`). Arming the dense FFN must not arm these;
///   that separation is the whole point of the scoped lever (#917's 10.3 GiB).
/// * `!disabled` — the `ATLAS_NO_W8A8_DECODE_PROJ` kill switch.
/// * `DECODE_W8A8_ROWS.contains(&rows)` — the 5..=16 band, on the PADDED n.
/// * `Fp8BlockScaled` — cuBLASLt is told the weight scales are a BLK128x128
///   grid; a per-row `row_scale` has a different shape and reads as garbage.
/// * `n % 128 == 0`, `k % 128 == 0` — that grid is `[N/128, K/128]` and the
///   activation quantizer emits one scale per 128-wide K group.
/// * `blk128x128_stride_ok(k)` (i.e. `k % 512 == 0`) — cuBLAS requires the
///   weight-scale column stride `K/128` to be a multiple of 4.
/// * `ldc >= n` — a leading dimension shorter than the column is rejected by
///   the library; for a contiguous output the caller passes `ldc = n`.
/// * OUTPUT ROOM for the full padded write extent (see
///   [`strided_out_extent_elems`]).
/// * the quantizer kernel, and the VEC128 scale-layout adapter (kernel AND
///   scratch AND its capacity, and the FP8/scale scratch capacities). cuBLASLt
///   reads the activation scales token-contiguous; handing over the
///   quantizer's `[M, K/128]` order is fast and WRONG (H100 2026-09-11:
///   rel_rms 7.7e-2 / ~33 000 BF16 ULP on identical FP8 bytes). Falling back
///   to the batch16 GEMV is the only safe answer when any of it is missing.
#[derive(Clone, Copy, Debug)]
pub struct DecodeW8a8Plan {
    /// Padded decode rows — the ctx `n` the CUDA-graph cache is keyed by.
    pub rows: usize,
    /// Output width of this projection.
    pub n: u32,
    /// Contract width of this projection.
    pub k: u32,
    /// Output row pitch in BF16 ELEMENTS (`n` for a contiguous output).
    pub ldc: u32,
    /// Allocated size of the output buffer, in bytes.
    pub out_capacity_bytes: usize,
}

impl DecodeW8a8Plan {
    /// A contiguous `[rows, n]` output (SSM `in_proj_qkvz`, SSM `out_proj`,
    /// attention `o_proj`).
    pub fn contiguous(rows: usize, n: u32, k: u32, out_capacity_bytes: usize) -> Self {
        Self {
            rows,
            n,
            k,
            ldc: n,
            out_capacity_bytes,
        }
    }

    /// A strided output: rows `ldc` BF16 elements apart (attention Q/K/V into
    /// the `[n, per_seq_qkv]` multi-seq QKV buffer).
    pub fn strided(rows: usize, n: u32, k: u32, ldc: u32, out_capacity_bytes: usize) -> Self {
        Self {
            rows,
            n,
            k,
            ldc,
            out_capacity_bytes,
        }
    }

    /// The padded M cuBLASLt is actually handed.
    pub fn m_pad(&self) -> u32 {
        cublas_fp8_m_pad(self.rows as u32)
    }

    /// Bytes of `out` this projection may touch, phantom rows included.
    pub fn write_extent_bytes(&self) -> usize {
        strided_out_extent_elems(self.m_pad(), self.ldc, self.n) * 2
    }
}

/// The activation-quant scratch triple + its capacities, read straight off the
/// buffer arena at the call site. Bundled so the selector takes one argument
/// instead of six and the tests can build it without an arena.
#[derive(Clone, Copy, Debug)]
pub struct DecodeW8a8Scratch {
    pub act_fp8: DevicePtr,
    pub act_fp8_bytes: usize,
    pub act_scale: DevicePtr,
    pub act_scale_bytes: usize,
    pub act_scale_kmajor: DevicePtr,
    pub act_scale_kmajor_bytes: usize,
    pub quant_k: Fp8ActQuant,
    pub scale_kmajor_k: KernelHandle,
}

impl DecodeW8a8Scratch {
    /// Whether the scratch can hold `m_pad` rows of a K-wide activation.
    fn fits(&self, m_pad: u32, k: u32) -> bool {
        let rows = m_pad as usize;
        let kg = k as usize / 128;
        self.act_fp8.0 != 0
            && self.act_scale.0 != 0
            && self.quant_k.available()
            && self.act_fp8_bytes >= rows * k as usize
            && self.act_scale_bytes >= rows * kg * 4
            && (!cublas_scale_layout_kmajor()
                || (self.scale_kmajor_k.0 != 0
                    && self.act_scale_kmajor.0 != 0
                    && self.act_scale_kmajor_bytes >= rows * kg * 4))
    }
}

/// Whether ONE decode projection takes the W8A8 cuBLASLt arm. See
/// [`DecodeW8a8Plan`] for the clause-by-clause rationale.
pub fn decode_w8a8_selected(
    family_armed: bool,
    disabled: bool,
    plan: &DecodeW8a8Plan,
    scale_format: crate::weight_map::WeightQuantFormat,
    scratch: &DecodeW8a8Scratch,
) -> bool {
    let m_pad = plan.m_pad();
    family_armed
        && !disabled
        && DECODE_W8A8_ROWS.contains(&plan.rows)
        && scale_format == crate::weight_map::WeightQuantFormat::Fp8BlockScaled
        && plan.n.is_multiple_of(128)
        && plan.k.is_multiple_of(128)
        && spark_runtime::cublaslt::scale_layout::blk128x128_stride_ok(plan.k as usize)
        && plan.ldc >= plan.n
        && plan.write_extent_bytes() <= plan.out_capacity_bytes
        && scratch.fits(m_pad, plan.k)
}

/// Quantize `act[rows, k]` BF16 ONCE into the shared scratch: FP8 E4M3 bytes +
/// per-token 1×128 FP32 scales, the phantom rows `rows..ceil16(rows)` zeroed,
/// and the VEC128 scales re-laid-out K-major for cuBLASLt.
///
/// Split from the GEMM so a caller with several projections over the SAME
/// activation pays it once — the attention layer's Q/K/V share `normed`, so
/// quantizing inside the GEMM helper would run the quantizer and the scale
/// transpose three times per layer per step. Exactly the split
/// `dense_ffn_w8a8_prefill` makes for gate/up.
///
/// A zero scale kills the phantom rows' CONTRIBUTION, but the FP8 dot product
/// still runs over whatever bytes are there and `NaN * 0.0` is `NaN` — hence
/// the memset of the FP8 bytes and not only of the scales.
pub fn decode_w8a8_quant_act(
    gpu: &dyn GpuBackend,
    scratch: &DecodeW8a8Scratch,
    act_bf16: DevicePtr,
    rows: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    per_token_group_quant_fp8(
        gpu,
        scratch.quant_k,
        act_bf16,
        scratch.act_fp8,
        scratch.act_scale,
        rows,
        k,
        stream,
    )?;
    let m_pad = cublas_fp8_m_pad(rows);
    if m_pad > rows {
        gpu.memset_async(
            scratch.act_fp8.offset(rows as usize * k as usize),
            0,
            (m_pad - rows) as usize * k as usize,
            stream,
        )?;
    }
    if cublas_scale_layout_kmajor() {
        // Writes every [K/128, m_pad] slot, pad rows included.
        fp8_act_scale_to_kmajor(
            gpu,
            scratch.scale_kmajor_k,
            scratch.act_scale,
            scratch.act_scale_kmajor,
            rows,
            m_pad,
            k,
            stream,
        )?;
    } else {
        // Measurement control only (`ATLAS_CUBLAS_SCALE_LAYOUT=rowmajor`): the
        // pad rows are a contiguous tail in THIS layout, so zero them here.
        let kg = k as usize / 128;
        if m_pad > rows {
            gpu.memset_async(
                scratch.act_scale.offset(rows as usize * kg * 4),
                0,
                (m_pad - rows) as usize * kg * 4,
                stream,
            )?;
        }
    }
    Ok(())
}

/// `out[rows, n] = act_fp8[rows, k] @ weight[n, k]ᵀ` at row pitch `plan.ldc`,
/// both block-scale sets folded in an FP32 epilogue.
///
/// The activation must already be through [`decode_w8a8_quant_act`]; this is
/// the GEMM alone, so N projections over one activation cost one quantize and
/// N matmuls.
pub fn decode_w8a8_gemm(
    scratch: &DecodeW8a8Scratch,
    fp8w: &crate::weight_map::Fp8Weight,
    out: DevicePtr,
    plan: &DecodeW8a8Plan,
    stream: u64,
) -> Result<()> {
    let b_scale = if cublas_scale_layout_kmajor() {
        scratch.act_scale_kmajor
    } else {
        scratch.act_scale
    };
    spark_runtime::cublaslt::fp8_gemm_act_weight_t_blkscaled_ldc(
        scratch.act_fp8.0,
        b_scale.0,
        fp8w.weight.0,
        fp8w.row_scale.0,
        out.0,
        plan.m_pad(),
        plan.n,
        plan.k,
        plan.ldc,
        stream,
    )
}

#[cfg(test)]
#[path = "dispatch_proj_decode_tests.rs"]
mod tests;
