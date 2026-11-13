// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the `lm_head_verify_gemv` row: a BF16 LM head's
//! speculative-verify rows above 8 on `dense_gemv_bf16_batchm`, 16 rows a
//! weight pass, instead of the scalar `dense_gemm`. Off by default on every
//! target.
//!
//! # What it replaces
//!
//! `TransformerModel::lm_head_batched` (`model/impl_a3.rs`) projects the
//! verify rows of every speculative pass. For a BF16 head it runs the batched
//! GEMV at 2..=8 rows (`ops::DENSE_GEMV_BATCHM_DECODE_MAX_M`), and above that
//! nothing but the scalar `dense_gemm`: the wide arms that follow are NVFP4
//! only. A batched verify of 8 streams with one draft each is 16 rows, so on
//! the Qwen3.5-4B's tied BF16 head (248,070 x 2,560, 1.27 GB) the scalar
//! kernel computes 10.2 G multiply-adds per verify pass. The tier-S timing
//! on the RTX 4060 Ti 16 GB (2026-11-13) has the verify forward at 92.6 ms
//! for those 16 rows and 148 ms for 32, against a 40 ms decode step of 8
//! rows; the head and the GDN projections (`gdn_verify_gemv`) are the two
//! scalar GEMMs in it.
//!
//! The batched GEMV's kernel serves 16 rows a launch
//! (`ops::DENSE_GEMV_BATCHM_MAX_M`) and measures 6.8 ms on this head at 16
//! rows against 5.0 ms at 8 (L07 sweep, same card), so a 16-row verify pays
//! about 2 ms over the 8-row decode head instead of about 20.
//!
//! # Numerics
//!
//! Every row is bit-identical to the M = 1 decode head (`dense_gemv_bf16`),
//! which is also what verify rows 2..=8 compute today. Rows above 8 leave the
//! scalar tile GEMM's summation order for that one, so their logits move,
//! toward what plain decode computes.
//!
//! # Scope
//!
//! Only a BF16 head on one rank: an FP8, NVFP4 or packed head never reaches
//! the arm, and a vocab-parallel (sharded) head keeps its path. The decode
//! head's band (`lm_head_batchm_max`) is a different row and is not touched;
//! the frozen `DENSE_GEMV_BATCHM_DECODE_MAX_M` still decides rows 2..=8.

use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::weight_map::DenseWeight;

/// Is the `lm_head_verify_gemv` row on (`AVAROK_LM_HEAD_VERIFY_GEMV`)?
pub fn row() -> bool {
    super::port_rows::lm_head_verify_gemv::row()
}

/// Whether the row serves `rows` verify rows of a BF16 head with hidden size
/// `k`, as a pure function:
///
/// * `row_on`, the batched GEMV kernel present, and the head not sharded
///   across ranks;
/// * `rows` above the band the batched GEMV already serves (2..=8);
/// * `k % 8 == 0`: the kernel reads rows as 16-byte words.
pub fn claims(row_on: bool, kernel_present: bool, sharded: bool, rows: usize, k: usize) -> bool {
    row_on
        && kernel_present
        && !sharded
        && rows > super::DENSE_GEMV_BATCHM_DECODE_MAX_M as usize
        && k.is_multiple_of(8)
}

/// `logits[rows, v] = hidden[rows, h] @ weight[v, h]^T` on the batched GEMV,
/// [`super::gdn_verify_gemv::ROWS_PER_PASS`] rows a launch (the launcher the
/// GDN verify row uses; a head is one more `[N, K]` BF16 weight).
#[allow(clippy::too_many_arguments)]
pub fn launch_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    weight: &DenseWeight,
    logits: DevicePtr,
    rows: usize,
    v: u32,
    h: u32,
    stream: u64,
) -> anyhow::Result<()> {
    super::gdn_verify_gemv::launch_rows(gpu, kernel, hidden, weight, logits, rows, v, h, stream)
}

#[cfg(test)]
#[path = "lm_head_verify_gemv_tests.rs"]
mod tests;
