// SPDX-License-Identifier: AGPL-3.0-only

//! The `ffn_bf16_cublas_prefill` row: the BF16 dense-FFN prefill on the
//! cuBLASLt BF16 GEMM by target default, without `AVAROK_CUBLAS_GEMM`.
//!
//! # Why a row and not a recipe
//!
//! On the RTX 4060 Ti 16 GB (Ada, sm_89) the hand-written `dense_gemm_tc` is
//! very slow on the dense-FFN prefill shapes. An end-to-end A/B on the card,
//! same binary and flags, one env change (`AVAROK_CUBLAS_GEMM=ffn` against
//! unset), 3,921-token prompt: Qwen3.5-0.8B prefill 8,199 -> 13,113 tok/s and
//! Qwen3.5-4B 795 -> 3,677 tok/s, decode unchanged. Same numerics class: BF16
//! in, FP32 accumulate. `kernels/ada/HARDWARE.toml` declares it ON; every
//! other target declares it OFF.
//!
//! # Why narrower than `AVAROK_CUBLAS_GEMM=ffn`
//!
//! `CublasScope::ffn` also arms the W8A8 dense-FFN prefill
//! (`layers/dense_ffn_w8a8_prefill.rs`) and the MoE shared expert
//! (`layers/moe/helpers_c.rs`). The receipt above measured neither, so the
//! row arms ONLY the `bf16_weights` branch of `DenseFfnLayer`'s prefill; the
//! FP8 route keeps its own kernels.
//!
//! # Resolution order
//!
//! 1. `AVAROK_CUBLAS_GEMM` set, to ANY value (`off` and empty included): the
//!    operator's scope decides exactly as before this row existed, and the row
//!    resolves to that scope's `ffn` bit, tagged `(env)`.
//! 2. `AVAROK_FFN_BF16_CUBLAS_PREFILL`: the row's own A/B, under the 2026-09-11
//!    toggle grammar (`0`/`false`/`off`/`no` = off, anything else = on).
//! 3. The target's declaration.
//!
//! # Without cuBLASLt
//!
//! No fallback is added here, deliberately: the arm calls the same
//! `ops::cublas_bf16_proj_dense` that `AVAROK_CUBLAS_GEMM=ffn` calls. The CUDA
//! build links `libcublasLt` at load time, so a host without the library fails
//! to start the binary at all; a library that loads but cannot create a handle
//! or find an algorithm returns an `Err` from the GEMM, which fails the
//! prefill with that message. `AVAROK_FFN_BF16_CUBLAS_PREFILL=0` (or
//! `AVAROK_CUBLAS_GEMM=off`) returns the layer to `dense_gemm_tc`.

use super::dispatch_config::parse_cublas_scope;
use super::target_defaults::{Resolved, Source};

/// The row's own override variable.
pub const ENV: &str = "AVAROK_FFN_BF16_CUBLAS_PREFILL";

/// Resolve the row from the target's declaration and the two variables, as a
/// pure function. `cublas_gemm` is `AVAROK_CUBLAS_GEMM` (`None` = absent);
/// `row` is [`ENV`].
pub fn resolve(default_on: bool, cublas_gemm: Option<&str>, row: Option<&str>) -> Resolved<bool> {
    if let Some(raw) = cublas_gemm {
        return Resolved {
            value: parse_cublas_scope(Some(raw)).0.ffn,
            source: Source::Env,
        };
    }
    super::target_defaults::resolve_toggle(default_on, row, false)
}

/// Whether the BF16 dense-FFN prefill takes the cuBLASLt arm.
///
/// `scope_ffn` is the layer's carried `CublasScope::ffn` bit (read in
/// `layers/dense_ffn.rs`, the family's home); `row` is the resolved row. When
/// `AVAROK_CUBLAS_GEMM` is set the row equals `scope_ffn` (see [`resolve`]),
/// so the answer is exactly the pre-row one; when it is absent `scope_ffn` is
/// false and the row decides.
pub fn takes_cublas(scope_ffn: bool, row: bool) -> bool {
    scope_ffn || row
}

/// The process-wide resolved row (`target_defaults::resolved()`).
pub fn row() -> bool {
    super::target_defaults::resolved()
        .ffn_bf16_cublas_prefill
        .value
}

#[cfg(test)]
#[path = "ffn_bf16_cublas_tests.rs"]
mod tests;
