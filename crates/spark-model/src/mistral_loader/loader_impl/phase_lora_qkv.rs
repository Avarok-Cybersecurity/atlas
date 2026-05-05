// SPDX-License-Identifier: AGPL-3.0-only

//! Phase A: load LoRA wq_a / wq_b / wkv_a / wkv_b, NVFP4 quantize, TP shard.

use anyhow::Result;

use super::ctx::MistralLayerCtx;
use crate::tp_shard::{TpShardKind, shard_dense_bf16};
use crate::weight_map::{DenseWeight, dense, quantize_to_nvfp4};

pub(super) fn load_lora_qkv(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    let ap = ctx.ap();
    let h = ctx.h;
    let q_lora = ctx.q_lora;
    let kv_lora = ctx.kv_lora;
    let nope = ctx.nope;
    let rope = ctx.rope;
    let v_dim = ctx.v_dim;
    let n_heads = ctx.n_heads;
    let n_kv = ctx.n_kv;
    let hd = ctx.hd;
    let bf16 = ctx.bf16;
    let gpu = ctx.gpu;

    // ── MLA 2-step decode: store LoRA components separately ──
    // Runtime NVFP4 quantization of the MLA projection weights gives a
    // ~1.5× decode speedup on GB10. The RMS norm immediately after each
    // projection dampens NVFP4 errors before caching. Set
    // ATLAS_NVFP4_MLA=0 to force BF16.
    let wq_a_dense = dense(ctx.store, &format!("{ap}.wq_a.weight"))?;
    let wq_a_nvfp4 = Some(quantize_to_nvfp4(
        &wq_a_dense,
        q_lora,
        h,
        gpu,
        ctx.absmax_k,
        ctx.quantize_k,
        ctx.stream,
    )?);
    let mut wq_b = dense(ctx.store, &format!("{ap}.wq_b.weight"))?;

    // TP: wq_b is column-parallel on the heads axis. After main.rs's
    // head split, n_heads is already TP-local; reconstruct the full
    // pre-shard rows = n_heads_local * tp_size * hd.
    let tp_rank = ctx.config.tp_rank;
    let tp_size = ctx.config.tp_world_size.max(1);
    if tp_size > 1 {
        let full_rows = n_heads * tp_size * hd;
        let (sharded, _, _) = shard_dense_bf16(
            wq_b.weight,
            full_rows,
            q_lora,
            TpShardKind::ColumnParallel,
            tp_rank,
            tp_size,
            gpu,
        )?;
        if sharded != wq_b.weight {
            gpu.free(wq_b.weight)?;
        }
        wq_b.weight = sharded;
    }
    let wq_b_nvfp4 = Some(quantize_to_nvfp4(
        &wq_b,
        n_heads * hd,
        q_lora,
        gpu,
        ctx.absmax_k,
        ctx.quantize_k,
        ctx.stream,
    )?);
    let q_a_norm = dense(ctx.store, &format!("{ap}.q_a_norm.weight"))?;

    // wkv_a: [kv_lora+rope, h] — first kv_lora rows for latent,
    // last rope for K_rope.
    let wkv_a_dense = dense(ctx.store, &format!("{ap}.wkv_a_with_mqa.weight"))?;
    let wkv_a_nvfp4 = Some(quantize_to_nvfp4(
        &wkv_a_dense,
        kv_lora + rope,
        h,
        gpu,
        ctx.absmax_k,
        ctx.quantize_k,
        ctx.stream,
    )?);
    let wkv_a_rope_dense = DenseWeight {
        weight: wkv_a_dense.weight.offset(kv_lora * h * bf16),
    };
    let mut wkv_b = dense(ctx.store, &format!("{ap}.wkv_b.weight"))?;
    if tp_size > 1 {
        let full_rows = n_kv * tp_size * (nope + v_dim);
        let (sharded, _, _) = shard_dense_bf16(
            wkv_b.weight,
            full_rows,
            kv_lora,
            TpShardKind::ColumnParallel,
            tp_rank,
            tp_size,
            gpu,
        )?;
        if sharded != wkv_b.weight {
            gpu.free(wkv_b.weight)?;
        }
        wkv_b.weight = sharded;
    }
    let kv_a_norm = dense(ctx.store, &format!("{ap}.kv_a_norm.weight"))?;

    ctx.wq_a_dense = Some(wq_a_dense);
    ctx.wq_a_nvfp4 = wq_a_nvfp4;
    ctx.wq_b = Some(wq_b);
    ctx.wq_b_nvfp4 = wq_b_nvfp4;
    ctx.q_a_norm = Some(q_a_norm);
    ctx.wkv_a_dense = Some(wkv_a_dense);
    ctx.wkv_a_nvfp4 = wkv_a_nvfp4;
    ctx.wkv_a_rope_dense = Some(wkv_a_rope_dense);
    ctx.wkv_b = Some(wkv_b);
    ctx.kv_a_norm = Some(kv_a_norm);
    Ok(())
}
