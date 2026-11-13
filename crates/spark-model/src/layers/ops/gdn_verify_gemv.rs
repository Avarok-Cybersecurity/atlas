// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the `gdn_verify_gemv` row: the K-token (MTP verify) GDN
//! projections of a BF16-kept GDN layer on `dense_gemv_bf16_batchm`, one
//! weight pass per 16 rows, every row bit-identical to the M = 1 decode GEMV.
//! Off by default on every target.
//!
//! # What it replaces
//!
//! `Qwen3SsmLayer::decode_batched_inner` (`qwen3_ssm/trait_decode_batched.rs`)
//! serves every MTP verify pass, one sequence (K = drafts + 1 rows) or a
//! batch (R = sum of the sequences' rows). On a GDN layer whose projections
//! stay BF16 (the Qwen3.5 dense `Nvfp4Variant::Bf16Raw` load) it runs:
//!
//! | rows | `in_proj_qkvz`                           | `out_proj_dense`    |
//! |------|------------------------------------------|---------------------|
//! | 2    | `dense_gemv_batch2` (one weight pass)    | scalar `dense_gemm` |
//! | 3, 4 | one `dense_gemv` PER ROW (3 or 4 passes) | scalar `dense_gemm` |
//! | > 4  | scalar `dense_gemm` (16x16 FFMA tiles)   | scalar `dense_gemm` |
//!
//! Both slow arms show in the tier-S verify timing on the RTX 4060 Ti 16 GB
//! (Qwen3.5-4B, FP8 FFN, 2026-11-13). One stream, 4 rows: the verify forward
//! is 47.9 ms against a 27 ms decode step; three extra passes over a 63 MB
//! weight in 24 layers are 4.5 GB, 17 ms at the measured 260 GB/s, and the
//! scalar out_proj is most of the rest. Eight streams: 92.6 ms at 16 rows and
//! 148 ms at 32 against a 40 ms decode step of 8 rows, which is the scalar
//! `dense_gemm` at about 0.6 TFLOP/s on both projections (2.2 and 4.5 ms a
//! layer). That is why MTP loses at 8 streams: 13.4 tokens in 99 ms against 8
//! in 40.
//!
//! `dense_gemv_bf16_batchm` reads the weight once for up to
//! [`ROWS_PER_PASS`] rows and stays near the weight-bandwidth floor (L07
//! sweep, same card, the 4B `in_proj_qkvz` shape: 234 us at 1 row, 253 at 8,
//! 310 at 16; `out_proj` 82, 84, 102).
//!
//! # Numerics
//!
//! The batched GEMV keeps `dense_gemv_bf16`'s per-row K order and reduction
//! tree at every width up to its `MAX_M` (`ops::DENSE_GEMV_BATCHM_MAX_M`),
//! and the M = 1 decode forward runs both projections on `dense_gemv_bf16`
//! (`qwen3_ssm/ssm_forward.rs`). So with the row on a verify row's GDN
//! projections are the decode's, bit for bit:
//!
//! * `in_proj_qkvz` at 3 and 4 rows: unchanged bits (the per-row loop it
//!   replaces is that same GEMV);
//! * `in_proj_qkvz` above 4 rows and `out_proj_dense` at every width: the
//!   scalar tile GEMM's summation order is replaced by the decode order, so
//!   verify logits move, toward what plain decode computes.
//!
//! # Against `gdn_batched_gemm`
//!
//! That row makes the same two launches on cuBLASLt, which reassociates the
//! K reduction and is flat in the row count. This row wins where both are
//! on, up to one pass ([`ROWS_PER_PASS`] rows); above that the cuBLASLt row,
//! when on, keeps its single pass and otherwise this row runs
//! `ceil(rows / 16)` passes, still far below the scalar GEMM.
//!
//! # Kernel handle
//!
//! The GDN layer has no field for this kernel, so it is read through the
//! backend's op cache: primed at layer construction when the row is on
//! (`qwen3_ssm/init.rs`), a map read per call after that. Nothing is looked
//! up with the row off.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::weight_map::DenseWeight;

/// Module and entry of the batched BF16 GEMV.
pub const KERNEL: (&str, &str) = ("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");

/// Rows one launch serves: the kernel's compile-time `MAX_M`.
pub const ROWS_PER_PASS: usize = super::DENSE_GEMV_BATCHM_MAX_M as usize;

/// Is the `gdn_verify_gemv` row on (`AVAROK_GDN_VERIFY_GEMV`)?
pub fn row() -> bool {
    super::port_rows::gdn_verify_gemv::row()
}

/// The batched GEMV's handle through the op cache, `0` where the target did
/// not compile the module (no lookup is issued then).
pub fn kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if !gpu.has_module(KERNEL.0) {
        return KernelHandle(0);
    }
    gpu.op_cache()
        .kernel(gpu, KERNEL.0, KERNEL.1)
        .unwrap_or(KernelHandle(0))
}

/// Construction-time lookup, so the forward never issues a first lookup.
/// Does nothing with the row off.
pub fn prime(gpu: &dyn GpuBackend) {
    if row() {
        let _ = kernel(gpu);
    }
}

/// Whether the row serves `rows` verify rows of a projection with reduction
/// depth `k`, as a pure function so every clause is pinned on the host:
///
/// * `row_on` and `kernel_present`;
/// * `rows >= min_rows`: 3 for `in_proj_qkvz` (two rows keep
///   `dense_gemv_batch2`, already one pass and bit-identical to decode), 2
///   for `out_proj_dense`;
/// * `k % 8 == 0`: the kernel reads each activation and weight row as
///   16-byte words;
/// * more than one pass only when `cublas_row` (the `gdn_batched_gemm` row)
///   is off, since cuBLASLt reads the weight once at any width.
pub fn claims(
    row_on: bool,
    cublas_row: bool,
    kernel_present: bool,
    rows: usize,
    min_rows: usize,
    k: usize,
) -> bool {
    row_on
        && kernel_present
        && rows >= min_rows
        && k.is_multiple_of(8)
        && (rows <= ROWS_PER_PASS || !cublas_row)
}

/// [`claims`] for `in_proj_qkvz` over the process rows and this backend.
pub fn qkvz_claims(gpu: &dyn GpuBackend, rows: usize, k: usize) -> bool {
    row()
        && claims(
            true,
            super::gdn_batched_gemm::row(),
            kernel(gpu).0 != 0,
            rows,
            3,
            k,
        )
}

/// [`claims`] for `out_proj_dense` over the process rows and this backend.
pub fn out_proj_claims(gpu: &dyn GpuBackend, rows: usize, k: usize) -> bool {
    row()
        && claims(
            true,
            super::gdn_batched_gemm::row(),
            kernel(gpu).0 != 0,
            rows,
            2,
            k,
        )
}

/// `(first_row, rows)` of each launch for `rows` rows in total.
pub fn passes(rows: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows)
        .step_by(ROWS_PER_PASS)
        .map(move |first| (first, (rows - first).min(ROWS_PER_PASS)))
}

/// `output[rows, n] = input[rows, k] @ weight[n, k]^T` on the batched GEMV,
/// [`ROWS_PER_PASS`] rows a launch. `input` and `output` are contiguous
/// row-major BF16.
#[allow(clippy::too_many_arguments)]
pub fn launch_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    rows: usize,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    const BF16: usize = 2;
    for (first, m) in passes(rows) {
        super::dense_gemv_batchm(
            gpu,
            kernel,
            input.offset(first * k as usize * BF16),
            weight,
            output.offset(first * n as usize * BF16),
            m as u32,
            n,
            k,
            n,
            stream,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "gdn_verify_gemv_tests.rs"]
mod tests;
