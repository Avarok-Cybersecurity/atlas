// SPDX-License-Identifier: AGPL-3.0-only

//! The `gdn_batched_gemm` row: the K-token (MTP verify) GDN projections of a
//! BF16-kept GDN layer through ONE cuBLASLt BF16 GEMM over all rows.
//!
//! # What it replaces
//!
//! `Qwen3SsmLayer::decode_batched_inner` (`qwen3_ssm/trait_decode_batched.rs`)
//! serves every MTP verify pass, single-sequence (K = drafts + 1 rows) and
//! batched (R = sum of ks rows). On a GDN layer whose projections stay BF16
//! (the Qwen3.5 dense `Nvfp4Variant::Bf16Raw` load: `in_proj_qkvz` and
//! `out_proj_dense` are the only live copies) it dispatched:
//!
//! | rows | `in_proj_qkvz`                          | `out_proj_dense`  |
//! |------|-----------------------------------------|-------------------|
//! | 2    | `dense_gemv_batch2` (one weight pass)   | scalar `dense_gemm` |
//! | 3, 4 | one `dense_gemv` PER ROW (3 or 4 passes) | scalar `dense_gemm` |
//! | > 4  | scalar `dense_gemm` (16x16 FFMA tiles)  | scalar `dense_gemm` |
//!
//! The per-row arm re-reads the 63 MB (4B: 12288 x 2560 BF16) weight once per
//! row, and the scalar `dense_gemm` streams at about 89 GB/s where a GEMV
//! reaches about 274 GB/s (measured, `multi_seq/attn/o_proj.rs`). The
//! multi-sequence DECODE path already moved both projections to
//! `ops::cublas_bf16_proj_dense` (`trait_decode_multi_seq/ssm_batched_proj.rs`:
//! qkvz 1.03 ms -> 381 us, out_proj 513 -> 194 us per layer at n = 2 on GB10).
//! With the row on, the verify path makes the same call:
//!
//! * `in_proj_qkvz` at 3 or more rows. Two rows keep `dense_gemv_batch2`,
//!   which already reads the weight once and is bit-identical to the M = 1
//!   decode GEMV.
//! * `out_proj_dense` at 2 or more rows.
//!
//! # Why OFF everywhere
//!
//! cuBLASLt reassociates the K reduction, so the verify rows stop matching
//! today's bits (at 3 and 4 rows they matched the M = 1 decode GEMV exactly).
//! A speculative pass whose verify logits move can accept a different token
//! than plain decoding would. That is a numerics change with no receipt yet,
//! so every target declares `false` and `AVAROK_GDN_BATCHED_GEMM=1` is the A/B.
//! The out_proj half already differed from decode (scalar tile GEMM against
//! the decode GEMV), so it trades one non-decode summation order for another.
//!
//! # Resolution order
//!
//! `AVAROK_GDN_BATCHED_GEMM` under the 2026-09-11 toggle grammar (`0`/`false`/
//! `off`/`no` = off, anything else = on), else the target's declaration.

use super::target_defaults::Resolved;

/// The row's override variable.
pub const ENV: &str = "AVAROK_GDN_BATCHED_GEMM";

/// Resolve the row from the declaration and [`ENV`] (`None` = absent).
pub fn resolve(default_on: bool, row: Option<&str>) -> Resolved<bool> {
    super::target_defaults::resolve_toggle(default_on, row, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    super::target_defaults::resolved().gdn_batched_gemm.value
}

/// Which kernel serves a BF16-only GDN `in_proj_qkvz` at `rows` verify rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkvzArm {
    /// `dense_gemv_batch2`: one weight pass, bit-identical to decode.
    GemvBatch2,
    /// One `dense_gemv` per row: `rows` weight passes.
    GemvPerRow,
    /// Scalar `dense_gemm` over all rows.
    DenseGemm,
    /// `cublas_bf16_proj_dense` over all rows.
    Cublas,
}

/// Which kernel serves a BF16 GDN `out_proj_dense` at `rows` verify rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutProjArm {
    DenseGemm,
    Cublas,
}

/// The QKVZ dispatch for a BF16-only layer. `row` off reproduces the arms
/// that existed before the row, exactly.
pub fn qkvz_arm(row: bool, rows: usize) -> QkvzArm {
    match rows {
        2 => QkvzArm::GemvBatch2,
        3.. if row => QkvzArm::Cublas,
        3 | 4 => QkvzArm::GemvPerRow,
        _ => QkvzArm::DenseGemm,
    }
}

/// The out_proj dispatch for a layer holding `out_proj_dense`.
pub fn out_proj_arm(row: bool, rows: usize) -> OutProjArm {
    if row && rows >= 2 {
        OutProjArm::Cublas
    } else {
        OutProjArm::DenseGemm
    }
}

#[cfg(test)]
#[path = "gdn_batched_gemm_tests.rs"]
mod tests;
