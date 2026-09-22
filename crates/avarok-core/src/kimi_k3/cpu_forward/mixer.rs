// SPDX-License-Identifier: AGPL-3.0-only
//! KDA / MLA mixers. Optional `gpu_gemv` runs resident BF16 matvecs.

use anyhow::Result;

use super::{Ablation, K3LayerCtx};
use crate::kimi_k3::attnres::rms_norm;
use crate::kimi_k3::cache::MlaKv;
use crate::kimi_k3::cpu_weights::{KdaWeights, MlaWeights};
use crate::kimi_k3::kda::{KdaConfig, KdaState, bounded_gate};
use crate::kimi_k3::mla::MlaConfig;
use crate::kimi_k3::ops::{matvec, matvec_column_tp};
use crate::kimi_k3::situ::sigmoid;

pub(super) fn gemv(
    ctx: &K3LayerCtx<'_>,
    op: &str,
    w: &[f32],
    x: &[f32],
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    if let Some(g) = ctx.gpu_gemv {
        return g(op, x, n, k);
    }
    Ok(matvec(w, x, n, k))
}

fn apply_o_proj(
    ctx: &K3LayerCtx<'_>,
    w: &[f32],
    x: &[f32],
    out: usize,
    inn: usize,
    ablation: Ablation,
) -> Result<Vec<f32>> {
    if ctx.gpu_gemv.is_some() {
        return gemv(ctx, "o_proj", w, x, out, inn);
    }
    Ok(matvec_column_tp(
        w,
        x,
        out,
        inn,
        ablation.o_proj_tp,
        ablation.drop_o_proj_rank,
    ))
}

pub(super) fn kda_mixer<F>(
    ctx: &K3LayerCtx<'_>,
    w: &KdaWeights,
    x: &[f32],
    cfg: &KdaConfig,
    state: &mut KdaState,
    eps: f32,
    ablation: Ablation,
    kda_decode: &mut F,
) -> Result<Vec<f32>>
where
    F: FnMut(&[f32], &[f32], &[f32], &[f32], &KdaConfig, &mut KdaState) -> Result<Vec<f32>>,
{
    let qdim = cfg.qkv_dim();
    let q = gemv(ctx, "q_proj", &w.q_proj, x, qdim, x.len())?;
    let k = gemv(ctx, "k_proj", &w.k_proj, x, qdim, x.len())?;
    let v = gemv(ctx, "v_proj", &w.v_proj, x, qdim, x.len())?;
    let mut qkv = q;
    qkv.extend_from_slice(&k);
    qkv.extend_from_slice(&v);
    let fa = gemv(ctx, "f_a_proj", &w.f_a, x, cfg.head_dim, x.len())?;
    let z = gemv(ctx, "f_b_proj", &w.f_b, &fa, qdim, cfg.head_dim)?;
    let gate = bounded_gate(
        &z,
        &w.dt_bias,
        &w.a_log,
        cfg.heads,
        cfg.head_dim,
        cfg.gate_lower_bound,
    );
    let beta = gemv(ctx, "b_proj", &w.b_proj, x, cfg.heads, x.len())?;
    let g = gemv(ctx, "g_proj", &w.g_proj, x, qdim, x.len())?;
    let core = kda_decode(&qkv, &w.conv, &gate, &beta, cfg, state)?;
    let gated = gated_o_norm(&core, &g, &w.o_norm, cfg.head_dim, eps);
    apply_o_proj(ctx, &w.o_proj, &gated, x.len(), qdim, ablation)
}

fn gated_o_norm(core: &[f32], g: &[f32], o_norm: &[f32], head_dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; core.len()];
    for ((c, gg), o) in core
        .chunks_exact(head_dim)
        .zip(g.chunks_exact(head_dim))
        .zip(out.chunks_exact_mut(head_dim))
    {
        let n = rms_norm(c, o_norm, eps);
        for i in 0..head_dim {
            o[i] = sigmoid(gg[i]) * n[i];
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mla_mixer<F>(
    ctx: &K3LayerCtx<'_>,
    w: &MlaWeights,
    x: &[f32],
    cfg: &MlaConfig,
    kv: &mut MlaKv,
    pos: usize,
    theta: f32,
    eps: f32,
    ablation: Ablation,
    mla_decode: &mut F,
) -> Result<Vec<f32>>
where
    F: FnMut(
        &mut [f32],
        &mut [f32],
        &[f32],
        &[f32],
        &mut MlaKv,
        &MlaConfig,
        usize,
        f32,
    ) -> Result<Vec<f32>>,
{
    let qk = cfg.qk_head_dim();
    let qa = gemv(ctx, "q_a_proj", &w.q_a, x, cfg.q_lora_rank, x.len())?;
    let qa = rms_norm(&qa, &w.q_a_ln, eps);
    let mut q = gemv(
        ctx,
        "q_b_proj",
        &w.q_b,
        &qa,
        cfg.heads * qk,
        cfg.q_lora_rank,
    )?;
    let kv_in = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
    let kv_lat = gemv(ctx, "kv_a_proj", &w.kv_a, x, kv_in, x.len())?;
    let (c, pe) = kv_lat.split_at(cfg.kv_lora_rank);
    let c = rms_norm(c, &w.kv_a_ln, eps);
    let (k, v) = mla_kv_from_split(&w.k_b, &w.v_b, &c, pe, cfg);
    let mut k = k;
    let g = gemv(
        ctx,
        "g_proj",
        &w.g_proj,
        x,
        cfg.heads * cfg.v_head_dim,
        x.len(),
    )?;
    let attn = mla_decode(&mut q, &mut k, &v, &g, kv, cfg, pos, theta)?;
    apply_o_proj(
        ctx,
        &w.o_proj,
        &attn,
        x.len(),
        cfg.heads * cfg.v_head_dim,
        ablation,
    )
}

fn mla_kv_from_split(
    k_b: &[f32],
    v_b: &[f32],
    c: &[f32],
    k_pe: &[f32],
    cfg: &MlaConfig,
) -> (Vec<f32>, Vec<f32>) {
    let heads = cfg.heads;
    let nope = cfg.qk_nope_head_dim;
    let rope = cfg.qk_rope_head_dim;
    let dv = cfg.v_head_dim;
    let lora = cfg.kv_lora_rank;
    let qk = nope + rope;
    let mut k = vec![0f32; heads * qk];
    let mut v = vec![0f32; heads * dv];
    for h in 0..heads {
        for d in 0..nope {
            let mut acc = 0.0f32;
            for l in 0..lora {
                acc += k_b[h * lora * nope + l * nope + d] * c[l];
            }
            k[h * qk + d] = acc;
        }
        if rope > 0 {
            let dst = h * qk + nope;
            k[dst..dst + rope].copy_from_slice(k_pe);
        }
        for d in 0..dv {
            let mut acc = 0.0f32;
            for l in 0..lora {
                acc += v_b[h * dv * lora + d * lora + l] * c[l];
            }
            v[h * dv + d] = acc;
        }
    }
    (k, v)
}

#[allow(dead_code)]
fn pack_mla_kv(kvb: &[f32], k_pe: &[f32], cfg: &MlaConfig) -> (Vec<f32>, Vec<f32>) {
    let nope = cfg.qk_nope_head_dim;
    let rope = cfg.qk_rope_head_dim;
    let dv = cfg.v_head_dim;
    let qk = nope + rope;
    let mut k = vec![0.0f32; cfg.heads * qk];
    let mut v = vec![0.0f32; cfg.heads * dv];
    let stride = nope + dv;
    for h in 0..cfg.heads {
        let src = &kvb[h * stride..(h + 1) * stride];
        let kd = &mut k[h * qk..(h + 1) * qk];
        kd[..nope].copy_from_slice(&src[..nope]);
        if rope > 0 {
            kd[nope..].copy_from_slice(k_pe);
        }
        v[h * dv..(h + 1) * dv].copy_from_slice(&src[nope..]);
    }
    (k, v)
}
