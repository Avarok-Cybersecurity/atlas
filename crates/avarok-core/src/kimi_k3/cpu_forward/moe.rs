// SPDX-License-Identifier: AGPL-3.0-only

//! Latent MoE host mix. Under TP the routed down/up matrices split **hidden**,
//! which is 7168/8 = 896. That 896 is not `n_experts`.

use super::{DenseMlp, K3LayerCtx};
use crate::kimi_k3::cpu_weights::MoeWeights;
use crate::kimi_k3::latent_moe::{LatentMoeConfig, sigmoid_topk};
use crate::kimi_k3::{attnres::rms_norm, cpu_forward::dense};
use anyhow::{Result, ensure};

pub(super) fn moe_mlp_with<F>(
    w: &MoeWeights,
    x: &[f32],
    ctx: &K3LayerCtx<'_>,
    force: Option<usize>,
    experts_fn: &mut F,
) -> Result<Vec<f32>>
where
    F: FnMut(&MoeWeights, &[f32], &[usize], &[f32], &LatentMoeConfig) -> Result<Vec<f32>>,
{
    let hidden = ctx.moe.hidden;
    let latent = ctx.moe.latent;
    ensure!(x.len() == hidden, "K3 MoE hidden {} vs {hidden}", x.len());
    let mut logits = super::mixer::gemv(ctx, "router", &w.router, x, ctx.moe.n_routed, hidden)?;
    if let Some(e) = force {
        logits.fill(0.0);
        logits[e] = 8.0;
    }
    let shared = w
        .shared
        .as_ref()
        .map(|s| shared_mlp(ctx, s, x))
        .transpose()?;
    let down_k = if !w.down.is_empty() {
        w.down.len() / latent
    } else {
        hidden / ctx.tp_world.max(1) // 896 at tp=8 is hidden/tp, not n_experts
    };
    let x_down = hidden_shard(x, down_k, ctx.tp_rank, ctx.tp_world)?;
    let mut lat = super::mixer::gemv(ctx, "routed_down", &w.down, x_down, latent, down_k)?;
    // Row-parallel down: each rank saw hidden/tp. 896 here is 7168/8, not experts.
    if down_k != hidden {
        reduce(ctx, &mut lat)?;
    }
    let (ids, weights) = sigmoid_topk(&logits, &w.bias, ctx.moe.top_k);
    let mixed = experts_fn(w, &lat, &ids, &weights, ctx.moe)?;
    let mixed = if ctx.moe.use_norm {
        rms_norm(&mixed, &w.norm, ctx.eps)
    } else {
        mixed
    };
    let up_n = if !w.up.is_empty() {
        w.up.len() / latent
    } else {
        hidden / ctx.tp_world.max(1) // 896 at tp=8 is hidden/tp, not n_experts
    };
    let routed = super::mixer::gemv(ctx, "routed_up", &w.up, &mixed, up_n, latent)?;
    let mut out = scatter_hidden(routed, hidden, ctx.tp_rank, ctx.tp_world)?;
    if let Some(s) = shared {
        ensure!(s.len() == hidden, "K3 shared expert width");
        for (o, ss) in out.iter_mut().zip(&s) {
            *o += *ss;
        }
    }
    if ctx.tp_world > 1 {
        reduce(ctx, &mut out)?;
    }
    Ok(out)
}

fn shared_mlp(ctx: &K3LayerCtx<'_>, s: &DenseMlp, x: &[f32]) -> Result<Vec<f32>> {
    let full = if ctx.shared_intermediate > 0 {
        ctx.shared_intermediate
    } else if !s.gate.is_empty() {
        ensure!(
            !x.is_empty() && s.gate.len().is_multiple_of(x.len()),
            "K3 shared gate geometry"
        );
        s.gate.len() / x.len() * ctx.tp_world.max(1)
    } else {
        anyhow::bail!(
            "K3 shared expert: set shared_intermediate or keep host gate weights (no silent 6144)"
        );
    };
    dense::run(ctx, s, x, full)
}

fn reduce(ctx: &K3LayerCtx<'_>, v: &mut [f32]) -> Result<()> {
    match ctx.reduce_hidden {
        Some(f) if ctx.tp_world > 1 => f(v),
        _ => Ok(()),
    }
}

fn hidden_shard(x: &[f32], local_k: usize, rank: usize, world: usize) -> Result<&[f32]> {
    if local_k == x.len() {
        return Ok(x);
    }
    ensure!(
        world > 1 && local_k * world == x.len() && rank < world,
        "K3 routed down K {local_k} vs hidden {} tp {world} rank {rank} (K is hidden/tp, not n_experts)",
        x.len()
    );
    let start = rank * local_k;
    Ok(&x[start..start + local_k])
}

fn scatter_hidden(partial: Vec<f32>, hidden: usize, rank: usize, world: usize) -> Result<Vec<f32>> {
    if partial.len() == hidden {
        return Ok(partial);
    }
    ensure!(
        world > 1 && partial.len() * world == hidden && rank < world,
        "K3 routed up N {} vs hidden {hidden} tp {world} rank {rank} (N is hidden/tp, not n_experts)",
        partial.len()
    );
    let mut full = vec![0.0f32; hidden];
    let start = rank * partial.len();
    full[start..start + partial.len()].copy_from_slice(&partial);
    Ok(full)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_over_tp_is_not_expert_count() {
        let hidden = 7168;
        let tp = 8;
        let local = hidden / tp;
        assert_eq!(local, 896);
        let x: Vec<f32> = (0..hidden).map(|i| i as f32).collect();
        let shard = hidden_shard(&x, local, 3, tp).unwrap();
        assert_eq!(shard.len(), 896);
        assert_eq!(shard[0], (3 * 896) as f32);
        let scattered = scatter_hidden(vec![1.0; 896], hidden, 3, tp).unwrap();
        assert_eq!(scattered.len(), 7168);
        assert_eq!(scattered[3 * 896], 1.0);
        assert_eq!(scattered[0], 0.0);
    }

    #[test]
    fn tp1_keeps_the_full_hidden() {
        let x = vec![1.0, 2.0, 3.0];
        assert_eq!(hidden_shard(&x, 3, 0, 1).unwrap(), x.as_slice());
        assert_eq!(scatter_hidden(x.clone(), 3, 0, 1).unwrap(), x);
    }

    #[test]
    fn sliced_down_up_ops_use_hidden_over_tp() {
        let model = crate::kimi_k3::cpu_weights::K3CpuModel::synthetic_small();
        let mut ctx = K3LayerCtx::from_model(&model);
        ctx.tp_rank = 1;
        ctx.tp_world = 2;
        let hidden = ctx.moe.hidden;
        let latent = ctx.moe.latent;
        let n_routed = ctx.moe.n_routed;
        let local = hidden / ctx.tp_world;
        assert_eq!(hidden, 8);
        assert_eq!(local, 4);
        assert_ne!(local, n_routed, "local K is hidden/tp, not n_experts");
        let w = MoeWeights {
            down: vec![0.01; latent * local],
            up: vec![0.01; local * latent],
            norm: vec![1.0; latent],
            router: vec![0.0; n_routed * hidden],
            bias: vec![0.0; n_routed],
            experts: vec![],
            shared: None,
        };
        let x = vec![1.0; hidden];
        let mut experts =
            |_w: &MoeWeights, lat: &[f32], _i: &[usize], _mw: &[f32], _c: &LatentMoeConfig| {
                Ok(vec![0.0; lat.len()])
            };
        let out = moe_mlp_with(&w, &x, &ctx, Some(0), &mut experts).unwrap();
        assert_eq!(out.len(), hidden);
        assert_eq!(w.down.len(), latent * local);
    }

    #[test]
    fn empty_shared_expert_without_configured_inter_does_not_invent_6144() {
        // Oracle: missing geometry must fail. Known-bad was a silent 6144 default.
        let model = crate::kimi_k3::cpu_weights::K3CpuModel::synthetic_small();
        let mut ctx = K3LayerCtx::from_model(&model);
        ctx.shared_intermediate = 0;
        let hidden = ctx.moe.hidden;
        let latent = ctx.moe.latent;
        let n_routed = ctx.moe.n_routed;
        let w = MoeWeights {
            down: vec![0.01; latent * hidden],
            up: vec![0.01; hidden * latent],
            norm: vec![1.0; latent],
            router: vec![0.0; n_routed * hidden],
            bias: vec![0.0; n_routed],
            experts: vec![],
            shared: Some(crate::kimi_k3::cpu_weights::DenseMlp {
                gate: vec![],
                up: vec![],
                down: vec![],
            }),
        };
        let x = vec![1.0; hidden];
        let mut experts =
            |_w: &MoeWeights, lat: &[f32], _i: &[usize], _mw: &[f32], _c: &LatentMoeConfig| {
                Ok(vec![0.0; lat.len()])
            };
        let err = moe_mlp_with(&w, &x, &ctx, Some(0), &mut experts).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("shared_intermediate") || msg.contains("shared expert"),
            "{msg}"
        );
    }
}
