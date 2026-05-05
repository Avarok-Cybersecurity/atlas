// SPDX-License-Identifier: AGPL-3.0-only

//! Phase B: per-head transpose of W_UK, W_UV; carve out wq_b_rope rows.

use anyhow::Result;

use super::super::gpu_alloc_or_managed;
use super::ctx::MistralLayerCtx;
use crate::weight_map::DenseWeight;

pub(super) fn build_per_head_views(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    let n_kv = ctx.n_kv;
    let kv_lora = ctx.kv_lora;
    let nope = ctx.nope;
    let rope = ctx.rope;
    let v_dim = ctx.v_dim;
    let hd = ctx.hd;
    let q_lora = ctx.q_lora;
    let bf16 = ctx.bf16;
    let gpu = ctx.gpu;
    let stride = nope + v_dim;
    let wkv_b = ctx.wkv_b.as_ref().expect("phase A must precede");
    let wq_b = ctx.wq_b.as_ref().expect("phase A must precede");

    // Q absorption: Q_absorbed[lkv] = sum_p(Q_nope[p] * wkv_b_k[lkv, p])
    // wkv_b has K_nope portion as [nope, kv_lora] per head — must
    // TRANSPOSE to [kv_lora, nope] for correct dot product. D2H the
    // relevant portion of wkv_b, transpose on CPU, upload.
    let wkv_b_total_rows = n_kv * stride;
    let wkv_b_bytes = wkv_b_total_rows * kv_lora * bf16;
    let mut wkv_b_host = vec![0u8; wkv_b_bytes];
    gpu.copy_d2h(wkv_b.weight, &mut wkv_b_host)?;

    // Transpose K portion: [nope, kv_lora] → [kv_lora, nope] per head.
    let w_uk_per_head = kv_lora * nope * bf16;
    let mut w_uk_host = vec![0u8; n_kv * w_uk_per_head];
    for head in 0..n_kv {
        for p in 0..nope {
            for lkv in 0..kv_lora {
                let src_off = ((head * stride + p) * kv_lora + lkv) * bf16;
                let dst_off = (head * kv_lora * nope + lkv * nope + p) * bf16;
                w_uk_host[dst_off..dst_off + bf16]
                    .copy_from_slice(&wkv_b_host[src_off..src_off + bf16]);
            }
        }
    }
    let w_uk_t_ptr = gpu_alloc_or_managed(gpu, n_kv * w_uk_per_head)?;
    gpu.copy_h2d(&w_uk_host, w_uk_t_ptr)?;

    // W_UV[n, l, v]: bmm-friendly extraction layout. We need:
    // attn_latent[N, 1, Lkv] @ W_UV[N, Lkv, V] → [N, 1, V]
    // For now store as [N, v_dim, kv_lora] and use a transposed-convention
    // GEMV path in V extraction (TODO: GPU transpose kernel).
    let w_uv_ptr = gpu_alloc_or_managed(gpu, n_kv * kv_lora * v_dim * bf16)?;
    for head in 0..n_kv {
        for v in 0..v_dim {
            let src_row = head * stride + nope + v;
            let src = wkv_b.weight.offset(src_row * kv_lora * bf16);
            let dst = w_uv_ptr.offset((head * v_dim * kv_lora + v * kv_lora) * bf16);
            gpu.copy_d2d(src, dst, kv_lora * bf16)?;
        }
    }

    // Extract wq_b_rope: the rope portion of wq_b per head.
    // wq_b_rope[n*rope+r, l] = wq_b[n*hd+nope+r, l] for r in 0..rope.
    let wqbr_ptr = gpu_alloc_or_managed(gpu, n_kv * rope * q_lora * bf16)?;
    for head in 0..n_kv {
        for r in 0..rope {
            let src_row = head * hd + nope + r;
            let src = wq_b.weight.offset(src_row * q_lora * bf16);
            let dst = wqbr_ptr.offset((head * rope + r) * q_lora * bf16);
            gpu.copy_d2d(src, dst, q_lora * bf16)?;
        }
    }

    ctx.wq_b_rope = Some(DenseWeight { weight: wqbr_ptr });
    ctx.w_uk_t = Some(DenseWeight { weight: w_uk_t_ptr });
    ctx.w_uv = Some(DenseWeight { weight: w_uv_ptr });
    ctx.w_uk_host = w_uk_host;
    Ok(())
}
