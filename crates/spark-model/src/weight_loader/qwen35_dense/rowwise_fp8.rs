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
//! Loads the checkpoint's per-row FP8 as-is, dequantises it ONCE to BF16, and
//! multiplies with cuBLASLt BF16. No re-quantisation anywhere: every FP8 E4M3
//! value is exactly representable in BF16, so the checkpoint's precision
//! survives intact, where the default path dequantises to BF16 and then
//! throws half of it away again by quantising to NVFP4.
//!
//! It was first written to keep FP8 all the way into a row-wise FP8 GEMM, and
//! that GEMM turns out not to work on this hardware — see the section below,
//! which is worth reading before trying it again.
//!
//! ## Prefill only, on purpose
//!
//! Decode is NOT wired here. `qkvz_fp8w` is read by `w8a16_gemv` in
//! `ssm_forward.rs` and `trait_decode_batched.rs`, and those are block-scaled;
//! putting a per-row weight in that field is exactly the misindexing described
//! above. So the row-wise weights live in their own fields, the NVFP4 copy is
//! still built and still serves decode, and prefill is the phase that stops
//! double-quantising. Mixing precision across phases is already the house
//! pattern — the native-FP8 SSM arm logs that its FP8 copy is "PREFILL ONLY"
//! and that decode + batched verify read the NVFP4 copy.
//!
//! A decode-side fix needs a per-row `w8a16_gemv` variant, which does not
//! exist yet; see `docs/fp8-rowwise-mixed-precision.md`.
//!
//! ## The FP8 GEMM is dead on GB10 — so the fold lands via BF16
//!
//! This was first written against `cublas_fp8_rowwise_proj`, keeping FP8 all
//! the way into the GEMM. That does not work on sm_121:
//! `cublaslt::fp8_gemm_act_weight_t_rowwise` declares both scales
//! `SCALE_MODE_OUTER_VEC_32F` and `cublasLtMatmulAlgoGetHeuristic` returns
//! status 15 (NOT_SUPPORTED). Padding M to 16 — which that call also needed,
//! and now does — does not change it.
//!
//! CONTROL, which makes that a statement about the GEMM and not about per-row
//! weights: the BLOCK-scaled `Qwen/Qwen3.8-27B-FP8` served with
//! `ATLAS_CUBLAS_FP8=1`, this module inert, reaches the same call through the
//! requant path and fails identically. Its sibling is worse —
//! `ATLAS_FP8_W8A8=1` passes the heuristic and returns "kililililil…". Both
//! sit behind default-off flags nothing in the repo sets, which is why the
//! whole cuBLASLt FP8 prefill family had never been noticed as dead code.
//!
//! So the weights are dequantised ONCE to BF16 and multiplied by cuBLASLt
//! BF16 instead. That is still the point of the fold: every FP8 E4M3 value is
//! exactly representable in BF16, so nothing is lost, where the default path
//! dequantises to BF16 and then throws half of it away again by quantising to
//! NVFP4. The same `dequant_fp8_blockscaled_bf16` kernel serves both layouts
//! — `block_n = 1, block_k = K, sk = 1` makes its index `scale[n]`.
//!
//! MEASURED on unsloth/Qwen3.8-27B-NVFP4, 2026-08-15, against the NVFP4
//! baseline on the same box and flags:
//!
//!   prefill      507 -> 585 tok/s   +15.5%
//!   decode       5.3 -> 5.3 tok/s    unchanged (decode is still NVFP4)
//!   dark-green probe   [red, blue] -> [red, blue, yellow]
//!   token match 82.5%, mean KL 0.0054, p99 0.039
//!   vision-fidelity 14/14 + 3/3, video-fidelity 13/13, both control held
//!
//! For contrast, `ATLAS_GDN_BF16_WEIGHTS=1` buys the same precision back
//! through the hand-written `dense_gemm` and costs 72.9% of prefill. The GEMM
//! is what separates them, not the precision.
//!
//! Keeping FP8 all the way — FP8 memory as well as FP8 precision — still
//! wants a per-row FP8 GEMM that works on this hardware, with a bit-parity
//! microtest in the shape of PR #474's. That is the remaining upside, not a
//! blocker.

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
