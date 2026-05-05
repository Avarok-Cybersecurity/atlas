// SPDX-License-Identifier: AGPL-3.0-only

//! Phase E: load output projection weight (BF16 + NVFP4 quantized).

use anyhow::Result;

use super::ctx::MistralLayerCtx;
use crate::weight_map::{dense, quantize_to_nvfp4};

pub(super) fn load_o_proj(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    let ap = ctx.ap();
    let h = ctx.h;
    let n_heads = ctx.n_heads;
    let hd = ctx.hd;
    let gpu = ctx.gpu;

    // O projection — NVFP4 quantized for decode throughput. The output
    // projection reads the attention output (which has RMS norm applied
    // upstream) and produces the residual contribution. NVFP4 of wo
    // alone accounts for ~15% of the MLA decode speedup.
    let o_dense_bf16 = dense(ctx.store, &format!("{ap}.wo.weight"))?;
    let o_nvfp4 = Some(quantize_to_nvfp4(
        &o_dense_bf16,
        h,
        n_heads * hd,
        gpu,
        ctx.absmax_k,
        ctx.quantize_k,
        ctx.stream,
    )?);

    ctx.o_dense_bf16 = Some(o_dense_bf16);
    ctx.o_nvfp4 = o_nvfp4;
    Ok(())
}
