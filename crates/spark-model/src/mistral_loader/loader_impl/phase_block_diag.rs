// SPDX-License-Identifier: AGPL-3.0-only

//! Phase D: assemble block-diagonal W_UK_BD and W_UV_BD for prefill
//! batched GEMM.

use anyhow::Result;

use super::super::gpu_alloc_or_managed;
use super::ctx::MistralLayerCtx;
use crate::weight_map::DenseWeight;

pub(super) fn build_block_diagonals(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    let n_kv = ctx.n_kv;
    let kv_lora = ctx.kv_lora;
    let nope = ctx.nope;
    let v_dim = ctx.v_dim;
    let bf16 = ctx.bf16;
    let gpu = ctx.gpu;

    // Block-diagonal W_UK for prefill batched GEMM.
    // Layout: [nq*kv_lora, nq*nope] with block h at
    // [h*kv_lora:(h+1)*kv_lora, h*nope:(h+1)*nope].
    let bd_rows = n_kv * kv_lora;
    let bd_cols = n_kv * nope;
    let bd_size = bd_rows * bd_cols * bf16;
    let mut w_uk_bd_host = vec![0u8; bd_size]; // zeros = block-diagonal padding
    let w_uk_host = &ctx.w_uk_host;
    for head in 0..n_kv {
        for lkv in 0..kv_lora {
            for p in 0..nope {
                let src_off = (head * kv_lora * nope + lkv * nope + p) * bf16;
                let dst_row = head * kv_lora + lkv;
                let dst_col = head * nope + p;
                let dst_off = (dst_row * bd_cols + dst_col) * bf16;
                w_uk_bd_host[dst_off..dst_off + bf16]
                    .copy_from_slice(&w_uk_host[src_off..src_off + bf16]);
            }
        }
    }
    let w_uk_bd_ptr = gpu_alloc_or_managed(gpu, bd_size)?;
    gpu.copy_h2d(&w_uk_bd_host, w_uk_bd_ptr)?;

    // Block-diagonal W_UV: [nq*v_dim, nq*kv_lora].
    let uv_bd_rows = n_kv * v_dim;
    let uv_bd_cols = n_kv * kv_lora;
    let uv_bd_size = uv_bd_rows * uv_bd_cols * bf16;
    let w_uv_ptr = ctx.w_uv.as_ref().expect("phase B").weight;
    let mut w_uv_host = vec![0u8; n_kv * v_dim * kv_lora * bf16];
    gpu.copy_d2h(w_uv_ptr, &mut w_uv_host)?;
    let mut w_uv_bd_host = vec![0u8; uv_bd_size];
    for head in 0..n_kv {
        for v in 0..v_dim {
            for l in 0..kv_lora {
                let src_off = (head * v_dim * kv_lora + v * kv_lora + l) * bf16;
                let dst_row = head * v_dim + v;
                let dst_col = head * kv_lora + l;
                let dst_off = (dst_row * uv_bd_cols + dst_col) * bf16;
                w_uv_bd_host[dst_off..dst_off + bf16]
                    .copy_from_slice(&w_uv_host[src_off..src_off + bf16]);
            }
        }
    }
    let w_uv_bd_ptr = gpu_alloc_or_managed(gpu, uv_bd_size)?;
    gpu.copy_h2d(&w_uv_bd_host, w_uv_bd_ptr)?;

    if ctx.layer_idx == 0 {
        tracing::info!(
            "MLA block-diagonal: W_UK [{},{}] ({:.1}MB), W_UV [{},{}] ({:.1}MB)",
            bd_rows,
            bd_cols,
            bd_size as f64 / 1e6,
            uv_bd_rows,
            uv_bd_cols,
            uv_bd_size as f64 / 1e6
        );
    }

    ctx.w_uk_block_diag = Some(DenseWeight {
        weight: w_uk_bd_ptr,
    });
    ctx.w_uv_block_diag = Some(DenseWeight {
        weight: w_uv_bd_ptr,
    });
    Ok(())
}
