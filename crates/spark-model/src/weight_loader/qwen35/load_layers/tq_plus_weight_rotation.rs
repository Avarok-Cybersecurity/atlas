// SPDX-License-Identifier: AGPL-3.0-only

//! TurboQuant+ weight pre-rotation.
//!
//! Folds the canonical S2·H·S1/sqrt(d) rotation into Q/K/V projection
//! weights at load time so the per-token `wht_bf16_inplace` dispatch on Q
//! and K/V can be skipped. For W_o, the inverse rotation R^T = S1·H·S2 is
//! folded into the input-side rows so the attention output (which is in
//! WHT-rotated V-basis) is brought back to original basis by o_proj alone
//! — no separate `wht_bf16_inplace_inv` launch needed.
//!
//! Per-token runtime savings (Qwen3.6-A3B at 40 attention layers):
//!   - Skip 1 wht_bf16_inplace on Q
//!   - Skip 1 wht_bf16_inplace on K
//!   - Skip 1 wht_bf16_inplace on V
//!   - Skip 1 wht_bf16_inplace_inv on attn output
//!
//! = 4 × 40 = 160 kernel launches saved per token
//!
//! Implementation: reuses the existing `wht_bf16_inplace` kernel. The
//! kernel processes `head_dim`-sized contiguous chunks indexed by
//! `blockIdx.x`, so a weight matrix laid out as `[hidden, n_heads * head_dim]`
//! can be rotated by treating each `head_dim` chunk as one "head" —
//! grid = `hidden * n_heads`, block = 32 (warp).
//!
//! For Q/K/V projections: weight rows are `[hidden, n_heads * head_dim]`
//! (column-major from a gemm viewpoint, or row-major if the loader
//! transposed). Either way, each head_dim chunk gets the canonical rotation.
//!
//! For O projection: weights are `[n_heads * head_dim, hidden]`. The
//! rotation is on the INPUT side per head, so the layout is identical to
//! Q/K/V but interpreted differently — same kernel applies.

use crate::layers::try_kernel;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

/// Pre-rotate a Q/K/V/O projection weight tensor in-place using the canonical
/// two-sided Rademacher WHT rotation. Identical sign masks as
/// `tq_plus_signs.cuh` — the same rotation applied at runtime by
/// `wht_bf16_inplace`.
///
/// Layout: `weight` points to bf16 storage shaped logically as
/// `[outer, n_heads * head_dim]`. Each contiguous `head_dim` chunk is rotated
/// independently (canonical per-head WHT).
#[allow(dead_code)]
pub fn apply_canonical_rotation_inplace(
    gpu: &dyn GpuBackend,
    weight_bf16: DevicePtr,
    outer: usize,
    n_heads: usize,
    head_dim: usize,
    stream: u64,
) -> Result<()> {
    let total_heads = outer
        .checked_mul(n_heads)
        .expect("outer * n_heads overflow");
    if total_heads == 0 {
        return Ok(());
    }
    // hd=128 has TQ+ signs vendored. Other sizes fall through unchanged
    // (kernel applies plain WHT — still a unitary rotation, but without the
    // canonical Gaussianization).
    if !(head_dim == 128 || head_dim == 256 || head_dim == 512) {
        anyhow::bail!(
            "apply_canonical_rotation_inplace: unsupported head_dim {head_dim} (need 128, 256, or 512)"
        );
    }

    let wht_kernel = try_kernel(gpu, "wht_bf16", "wht_bf16_inplace");
    if wht_kernel.0 == 0 {
        anyhow::bail!("wht_bf16_inplace kernel handle not available");
    }

    KernelLaunch::new(gpu, wht_kernel)
        .grid([total_heads as u32, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(weight_bf16)
        .arg_u32(head_dim as u32)
        .launch(stream)?;

    Ok(())
}

/// Should weight pre-rotation be active for this build? Opt-in via
/// `TQ_PLUS_WEIGHT_ROTATION=1` until correctness is validated.
#[allow(dead_code)]
pub fn weight_rotation_enabled() -> bool {
    std::env::var("TQ_PLUS_WEIGHT_ROTATION")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}
