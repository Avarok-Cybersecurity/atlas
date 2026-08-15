// SPDX-License-Identifier: AGPL-3.0-only

//! Native PER-ROW FP8 for the GDN projections of mixed-precision checkpoints.
//!
//! ## What this is for
//!
//! `unsloth/Qwen3.8-27B-NVFP4` (and its Qwen3.6 siblings, re-quantised
//! 2026-07-10) declare `format = mixed-precision`: the MLP is NVFP4, but
//! `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`
//! and `lm_head` ship as **FP8 E4M3 with a per-CHANNEL scale** — a `[N,1]`
//! tensor, one multiplier per output row.
//!
//! Atlas cannot feed that to its native FP8 kernels: the whole `w8a16` family
//! indexes `block_scale[n/128, k/128]` (`kernels/gb10/common/w8a16_gemv.cu`),
//! so a per-row buffer would hand 127 of every 128 rows another row's
//! multiplier. It is SMALLER than the grid the kernel indexes, so it reads
//! in-bounds garbage rather than faulting — which is why
//! `proj_is_fp8_any_scale` refuses it, correctly.
//!
//! The consequence is that those tensors take the fallback: dequantise to
//! BF16, then RE-quantise to NVFP4. Eight-bit weights served at four. The GDN
//! arm's own comment records what that cost the last time it was measured on a
//! checkpoint whose toolchain deliberately kept the SSM projections
//! high-precision — BFCL-ST non_live 85.4 → 76.6.
//!
//! ## What this does instead
//!
//! cuBLASLt has a row-wise FP8 GEMM, and Atlas already routes SSM prefill
//! through it (`ops::cublas_fp8_rowwise_proj`). It normally *manufactures* its
//! operand by converting a block-scaled weight — fp8 → bf16 → row-wise fp8.
//! A per-channel checkpoint is ALREADY that operand, so with the passthrough
//! in `dispatch_proj::rowwise_pair_passthrough` the conversion disappears and
//! the checkpoint's own bytes go straight to the GEMM.
//!
//! ## Prefill only, on purpose
//!
//! Decode is NOT wired here. `qkvz_fp8w` is read by `w8a16_gemv` in
//! `ssm_forward.rs` and `trait_decode_batched.rs`, and those are block-scaled;
//! putting a per-row weight in that field is exactly the misindexing described
//! above. So the row-wise weights live in their own fields, the NVFP4 copy is
//! still built and still serves decode, and prefill is the phase that stops
//! double-quantising. Mixing precision across phases is already the house
//! pattern — the native-FP8 SSM arm logs "NVFP4 kept as structural fallback
//! for decode batch paths".
//!
//! A decode-side fix needs a per-row `w8a16_gemv` variant, which does not
//! exist yet; see `docs/fp8-rowwise-mixed-precision.md`.
//!
//! ## ⚠ THE GEMM THIS DEPENDS ON DOES NOT WORK ON GB10 (measured 2026-08-15)
//!
//! `ATLAS_FP8_ROWWISE=1` currently makes prefill FAIL, and the fault is not
//! in this file. `cublas_fp8_rowwise_proj` ends in
//! `cublaslt::fp8_gemm_act_weight_t_rowwise`, which declares both scales
//! `SCALE_MODE_OUTER_VEC_32F`; on sm_121 `cublasLtMatmulAlgoGetHeuristic`
//! returns status 15 (NOT_SUPPORTED) and the request 400s. Padding M to 16
//! (which that call also needed, and now does) does not change it.
//!
//! CONTROL, which is what makes this a statement about the GEMM rather than
//! about per-row weights: serving the BLOCK-scaled `Qwen/Qwen3.8-27B-FP8`
//! with `ATLAS_CUBLAS_FP8=1` — this module inert, its own opt-in flag unset —
//! reaches the same call through the requant path and fails identically.
//!
//! Its sibling is no better. `ATLAS_FP8_W8A8=1` (block-scaled cuBLASLt,
//! `fp8_gemm_act_weight_t_blkscaled`) is ACCEPTED by the heuristic and then
//! returns degenerate output — "kililililil…" on a plain prompt.
//!
//! So the whole cuBLASLt FP8 prefill family is dead code on this box: one arm
//! errors, the other is silently wrong, and both sit behind default-off flags
//! that nothing in the repo sets (`ATLAS_CUBLAS_FP8` appears exactly once —
//! its own definition). That is why nobody had noticed.
//!
//! This module is kept, and kept OPT-IN, because the loader half is correct
//! and tested and it is what a working GEMM would plug into. Making the fold
//! real needs one of:
//!   * a per-row FP8 GEMM that works on sm_121 (own kernel, own bit-parity
//!     test in the shape of PR #474's microtest), or
//!   * dequantising the per-row FP8 to BF16 ONCE and using the cuBLASLt BF16
//!     GEMM — still no double-quant, and `dequant_fp8_blockscaled_to_bf16`
//!     already reads a `[N,1]` scale correctly when handed `block_n = 1`.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use crate::weight_map::{Fp8Weight, WeightQuantFormat};
use spark_runtime::weights::{WeightDtype, WeightStore};

/// Opt-in. Default OFF until the A/B and the gates say otherwise — this
/// changes the numerics of every GDN prefill projection on the checkpoints it
/// fires for.
pub(super) fn rowwise_fp8_enabled() -> bool {
    std::env::var("ATLAS_FP8_ROWWISE").as_deref() == Ok("1")
}

/// True when `{prefix}.weight` is FP8 E4M3 with a PER-ROW scale — `[N]` or
/// `[N,1]`, one multiplier per output row.
///
/// Deliberately the complement of `proj_is_fp8_any_scale`: that one accepts a
/// `[N/128, K/128]` block grid or a per-tensor scalar and refuses this shape;
/// this one accepts only this shape. A tensor cannot satisfy both, so the two
/// arms can never both claim a projection.
pub(super) fn proj_is_fp8_per_row(store: &WeightStore, prefix: &str) -> bool {
    let Ok(w) = store.get(&format!("{prefix}.weight")) else {
        return false;
    };
    if w.dtype != WeightDtype::FP8E4M3 || w.shape.len() != 2 {
        return false;
    }
    let Ok(s) = store.get(&format!("{prefix}.weight_scale")) else {
        return false;
    };
    scale_is_per_row(w.shape[0], &s.shape, s.num_elements())
}

/// The shape decision on its own: is `scale` one multiplier per output row of
/// an `[n, k]` weight?
///
/// Pure so the CPU-only CI can test it — this predicate is the thing standing
/// between a per-row buffer and a kernel that would index it as a block grid,
/// and that mistake does not fault, it returns plausible garbage.
pub(super) fn scale_is_per_row(n: usize, scale_shape: &[usize], scale_elems: usize) -> bool {
    // Exactly N elements, laid out as `[N]` or `[N,1]`. A per-tensor scalar
    // (1 element) and a `[N/128, K/128]` grid both fail on the element count,
    // so they stay with `proj_is_fp8_any_scale` and its block kernels.
    scale_elems == n && matches!(scale_shape.len(), 1 | 2) && scale_shape[0] == n
}

/// Load one per-row FP8 projection as an `Fp8Weight` tagged `Fp8PerRow`.
///
/// The scale is widened to F32 on the host when the checkpoint stores it as
/// BF16 (unsloth does), because the row-wise GEMM reads `[N]` f32.
pub(super) fn load_fp8_per_row(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<Fp8Weight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    let (n, k) = (w.shape[0], w.shape[1]);
    let s = store.get(&format!("{prefix}.weight_scale"))?;
    anyhow::ensure!(
        s.num_elements() == n,
        "{prefix}.weight_scale must hold exactly one scale per row ([N] or [N,1]); \
         got shape {:?} for a [{n}, {k}] weight",
        s.shape,
    );
    let row_scale = match s.dtype {
        WeightDtype::FP32 => s.ptr,
        WeightDtype::BF16 => {
            let mut bf16 = vec![0u8; n * 2];
            gpu.copy_d2h(s.ptr, &mut bf16)?;
            let mut f32s = vec![0u8; n * 4];
            for i in 0..n {
                let v = f32::from_bits(
                    (u16::from_le_bytes([bf16[i * 2], bf16[i * 2 + 1]]) as u32) << 16,
                );
                f32s[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            let p = gpu.alloc(n * 4)?;
            gpu.copy_h2d(&f32s, p)?;
            p
        }
        other => {
            anyhow::bail!("{prefix}.weight_scale: unsupported dtype {other:?} (want F32/BF16)")
        }
    };
    Ok(Fp8Weight {
        weight: w.ptr,
        row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    })
}

/// Concatenate two per-row FP8 weights along rows: `[a.n + b.n, k]`.
///
/// Per-row scales make this trivial in a way block grids do not — the
/// concatenated scale vector is just the two vectors end to end, with no
/// padding and no stride arithmetic, because a row's multiplier does not
/// depend on which 128-row block it lands in. (The block-scaled sibling,
/// `concat_fp8_block_scaled`, has to copy grid rows at the right stride; a
/// bug there is what CAUSAL-PATHWAY-AUDIT Bug #1 was.)
pub(super) fn concat_fp8_per_row(
    a: &Fp8Weight,
    b: &Fp8Weight,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<Fp8Weight> {
    anyhow::ensure!(
        a.scale_format == WeightQuantFormat::Fp8PerRow
            && b.scale_format == WeightQuantFormat::Fp8PerRow,
        "concat_fp8_per_row needs two Fp8PerRow weights, got {:?} and {:?}",
        a.scale_format,
        b.scale_format,
    );
    let (a_w, b_w) = (a.n as usize * k, b.n as usize * k);
    let weight = gpu.alloc(a_w + b_w)?;
    gpu.copy_d2d(a.weight, weight, a_w)?;
    gpu.copy_d2d(b.weight, weight.offset(a_w), b_w)?;
    let (a_s, b_s) = (a.n as usize * 4, b.n as usize * 4);
    let row_scale = gpu.alloc(a_s + b_s)?;
    gpu.copy_d2d(a.row_scale, row_scale, a_s)?;
    gpu.copy_d2d(b.row_scale, row_scale.offset(a_s), b_s)?;
    Ok(Fp8Weight {
        weight,
        row_scale,
        n: a.n + b.n,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    })
}

#[cfg(test)]
#[path = "rowwise_fp8_tests.rs"]
mod rowwise_fp8_tests;
