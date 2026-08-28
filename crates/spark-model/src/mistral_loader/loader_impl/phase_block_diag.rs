// SPDX-License-Identifier: AGPL-3.0-only

//! Phase D: assemble block-diagonal W_UK_BD and W_UV_BD for prefill
//! batched GEMM.

use anyhow::Result;

use super::super::gpu_alloc_or_managed;
use super::ctx::MistralLayerCtx;
use crate::weight_map::DenseWeight;

pub(crate) fn build_block_diagonals(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    // Load-time phase timer — see the note in `phase_per_head`.
    let t_phase = std::time::Instant::now();
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
    let t = std::time::Instant::now();
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
    let t_uk_scatter = t.elapsed();
    let t = std::time::Instant::now();
    let w_uk_bd_ptr = gpu_alloc_or_managed(gpu, bd_size)?;
    gpu.copy_h2d(&w_uk_bd_host, w_uk_bd_ptr)?;
    let t_uk_h2d = t.elapsed();

    // Block-diagonal W_UV: [nq*v_dim, nq*kv_lora].
    let uv_bd_rows = n_kv * v_dim;
    let uv_bd_cols = n_kv * kv_lora;
    let uv_bd_size = uv_bd_rows * uv_bd_cols * bf16;
    let w_uv_ptr = ctx.w_uv.as_ref().expect("phase B").weight;
    let mut w_uv_host = vec![0u8; n_kv * v_dim * kv_lora * bf16];
    let t = std::time::Instant::now();
    gpu.copy_d2h(w_uv_ptr, &mut w_uv_host)?;
    let t_uv_d2h = t.elapsed();
    let t = std::time::Instant::now();
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
    let t_uv_scatter = t.elapsed();
    let t = std::time::Instant::now();
    let w_uv_bd_ptr = gpu_alloc_or_managed(gpu, uv_bd_size)?;
    gpu.copy_h2d(&w_uv_bd_host, w_uv_bd_ptr)?;
    let t_uv_h2d = t.elapsed();

    tracing::info!(
        "MLA phase D (block-diagonals) L{}: total={:.1}ms | uk-scatter={:.1}ms, \
         uk h2d ({:.1} MB)={:.1}ms, uv d2h={:.1}ms, uv-scatter={:.1}ms, \
         uv h2d ({:.1} MB)={:.1}ms",
        ctx.layer_idx,
        t_phase.elapsed().as_secs_f64() * 1e3,
        t_uk_scatter.as_secs_f64() * 1e3,
        bd_size as f64 / 1e6,
        t_uk_h2d.as_secs_f64() * 1e3,
        t_uv_d2h.as_secs_f64() * 1e3,
        t_uv_scatter.as_secs_f64() * 1e3,
        uv_bd_size as f64 / 1e6,
        t_uv_h2d.as_secs_f64() * 1e3,
    );

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
