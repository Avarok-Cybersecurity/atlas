// SPDX-License-Identifier: AGPL-3.0-only

//! One-token K3 CPU forward: AttnRes + mixer + MLP, then logits.

use anyhow::Result;

use super::attnres::{attnres_mix, rms_norm};
use super::cache::{HybridCache, LayerCache, MlaKv};
use super::cpu_weights::{
    Ablation, DenseMlp, K3CpuLayer, K3CpuModel, KdaWeights, MixerW, MlaWeights, MlpW, MoeWeights,
};
use super::kda::{KdaConfig, KdaState, bounded_gate, kda_decode_token};
use super::latent_moe::{LatentMoeConfig, mix_routed_experts, sigmoid_topk};
use super::mla::{MlaConfig, mla_decode_token};
use super::ops::{embed_token, matvec, matvec_column_tp};
use super::situ::{sigmoid, situ_glu_vec};

/// Intra-block AttnRes stream (completed blocks + running partial).
///
/// Matches HF `KimiDecoderLayer._forward_attn_residual`: mix the incoming
/// prefix with already-archived blocks, then at `layer_idx % block_size == 0`
/// archive that incoming prefix and reset the intra-block sum. Layer 0
/// therefore archives the embedding as its own source, not `embed + mixer`.
#[derive(Clone, Debug)]
pub struct AttnResStream {
    pub completed: Vec<Vec<f32>>,
    pub partial: Vec<f32>,
    block_size: usize,
}

impl AttnResStream {
    pub fn new(hidden: usize, block_size: usize) -> Self {
        Self {
            completed: Vec::new(),
            partial: vec![0.0; hidden],
            block_size: block_size.max(1),
        }
    }

    fn sources(&self) -> Vec<Vec<f32>> {
        // sources[0] is the skip (current prefix). Mix=0 must return this,
        // not the first archived block.
        let mut s = vec![self.partial.clone()];
        s.extend(self.completed.iter().cloned());
        s
    }

    pub fn mix(&self, query: &[f32], norm_w: &[f32], eps: f32, mix: f32) -> Vec<f32> {
        attnres_mix(&self.sources(), query, norm_w, eps, mix)
    }

    fn add(&mut self, delta: &[f32]) {
        for (p, d) in self.partial.iter_mut().zip(delta) {
            *p += *d;
        }
    }

    fn archive_incoming_at_block_start(&mut self, layer_idx: usize) {
        if layer_idx % self.block_size == 0 {
            self.completed.push(self.partial.clone());
            self.partial.fill(0.0);
        }
    }
}

/// Per-layer geometry the GPU wrapper and `forward_token` share.
///
/// `K3BoundLayer::decode` copies hidden D2H, runs this math, copies H2D.
/// LinearAttention injects CUDA conv+recurrent unless `K3_CUDA_KDA=0`.
/// FullAttention injects CUDA gated-NoPE MLA unless `K3_CUDA_MLA=0`.
pub struct K3LayerCtx<'a> {
    pub kda: &'a KdaConfig,
    pub mla: &'a MlaConfig,
    pub moe: &'a LatentMoeConfig,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub hidden: usize,
    pub dense_intermediate: usize,
    pub eps: f32,
    pub rope_theta: f32,
}

impl<'a> K3LayerCtx<'a> {
    pub fn from_model(model: &'a K3CpuModel) -> Self {
        Self {
            kda: &model.kda,
            mla: &model.mla,
            moe: &model.moe,
            situ_beta: model.graph.situ_beta,
            situ_linear_beta: model.graph.situ_linear_beta,
            hidden: model.graph.hidden,
            dense_intermediate: model.dense_intermediate,
            eps: model.eps,
            rope_theta: model.rope_theta,
        }
    }
}

/// Embed one token and run every decoder layer at `pos`. Updates the hybrid cache.
/// AttnRes is per-token (layer depth), not carried across the sequence.
pub fn forward_token(
    model: &K3CpuModel,
    token: u32,
    pos: usize,
    cache: &mut HybridCache,
    ablation: Ablation,
) -> Vec<f32> {
    let embed = embed_token(&model.embed, token, model.graph.hidden, model.vocab);
    let mut stream = AttnResStream::new(model.graph.hidden, model.graph.attn_res_block_size);
    stream.partial.clone_from(&embed);
    let ctx = K3LayerCtx::from_model(model);
    for layer in &model.layers {
        if ablation.skip_layer == Some(layer.spec.index) {
            continue;
        }
        let mixer_state = &mut cache.layers[layer.spec.index];
        forward_one_layer(&ctx, layer, pos, mixer_state, &mut stream, ablation);
    }
    let h = stream.mix(
        &model.output_res_proj,
        &model.output_res_norm,
        model.eps,
        ablation.attnres_mix,
    );
    rms_norm(&h, &model.final_norm, model.eps)
}

/// LM head over a hidden vector. `W` is `[vocab, hidden]`.
pub fn logits(model: &K3CpuModel, h: &[f32]) -> Vec<f32> {
    matvec(&model.lm_head, h, model.vocab, model.graph.hidden)
}

/// One decoder layer: AttnRes + (KDA|MLA) + MLP. Mutates this layer's
/// [`LayerCache`] and the token's [`AttnResStream`].
///
/// Default cores are [`kda_decode_token`] / [`mla_decode_token`] / host
/// [`mix_routed_experts`]. Serve CUDA injects via [`forward_one_layer_with_cores`].
pub fn forward_one_layer(
    ctx: &K3LayerCtx<'_>,
    layer: &K3CpuLayer,
    pos: usize,
    mixer_state: &mut LayerCache,
    stream: &mut AttnResStream,
    ablation: Ablation,
) {
    forward_one_layer_with_cores(
        ctx,
        layer,
        pos,
        mixer_state,
        stream,
        ablation,
        cpu_kda_core,
        cpu_mla_core,
        cpu_moe_core,
    )
    .expect("K3 CPU mixer cores are infallible")
}

fn cpu_kda_core(
    x: &[f32],
    w: &[f32],
    g: &[f32],
    b: &[f32],
    cfg: &KdaConfig,
    st: &mut KdaState,
) -> Result<Vec<f32>> {
    Ok(kda_decode_token(x, w, g, b, cfg, st))
}

#[allow(clippy::too_many_arguments)]
fn cpu_mla_core(
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    g: &[f32],
    kv: &mut MlaKv,
    cfg: &MlaConfig,
    pos: usize,
    theta: f32,
) -> Result<Vec<f32>> {
    Ok(mla_decode_token(q, k, v, g, kv, cfg, pos, theta))
}

/// Same as [`forward_one_layer`], with a replaceable KDA conv+recurrent core.
///
/// Projections, MLA, MLP, and AttnRes stay on this CPU path.
#[allow(clippy::too_many_arguments)]
pub fn forward_one_layer_with_kda_decode<F>(
    ctx: &K3LayerCtx<'_>,
    layer: &K3CpuLayer,
    pos: usize,
    mixer_state: &mut LayerCache,
    stream: &mut AttnResStream,
    ablation: Ablation,
    kda_decode: F,
) -> Result<()>
where
    F: FnMut(&[f32], &[f32], &[f32], &[f32], &KdaConfig, &mut KdaState) -> Result<Vec<f32>>,
{
    forward_one_layer_with_cores(
        ctx,
        layer,
        pos,
        mixer_state,
        stream,
        ablation,
        kda_decode,
        cpu_mla_core,
        cpu_moe_core,
    )
}

/// Same as [`forward_one_layer`], with a replaceable MLA rope+SDPA+gate core.
///
/// Projections, KDA, MLP, and AttnRes stay on this CPU path.
#[allow(clippy::too_many_arguments)]
pub fn forward_one_layer_with_mla_decode<F>(
    ctx: &K3LayerCtx<'_>,
    layer: &K3CpuLayer,
    pos: usize,
    mixer_state: &mut LayerCache,
    stream: &mut AttnResStream,
    ablation: Ablation,
    mla_decode: F,
) -> Result<()>
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
    forward_one_layer_with_cores(
        ctx,
        layer,
        pos,
        mixer_state,
        stream,
        ablation,
        cpu_kda_core,
        mla_decode,
        cpu_moe_core,
    )
}

fn cpu_moe_core(
    w: &MoeWeights,
    latent: &[f32],
    ids: &[usize],
    mix_w: &[f32],
    cfg: &LatentMoeConfig,
) -> Result<Vec<f32>> {
    Ok(mix_routed_experts(latent, ids, mix_w, &w.experts, cfg))
}

/// Same as [`forward_one_layer`], with replaceable KDA / MLA / routed-expert cores.
///
/// Projections, AttnRes, router, down/up, shared experts, and SiTU mix stay here
/// unless the MoE callback replaces the expert GEMMs.
#[allow(clippy::too_many_arguments)]
pub fn forward_one_layer_with_cores<FK, FM, FE>(
    ctx: &K3LayerCtx<'_>,
    layer: &K3CpuLayer,
    pos: usize,
    mixer_state: &mut LayerCache,
    stream: &mut AttnResStream,
    ablation: Ablation,
    mut kda_decode: FK,
    mut mla_decode: FM,
    mut moe_experts: FE,
) -> Result<()>
where
    FK: FnMut(&[f32], &[f32], &[f32], &[f32], &KdaConfig, &mut KdaState) -> Result<Vec<f32>>,
    FM: FnMut(
        &mut [f32],
        &mut [f32],
        &[f32],
        &[f32],
        &mut MlaKv,
        &MlaConfig,
        usize,
        f32,
    ) -> Result<Vec<f32>>,
    FE: FnMut(&MoeWeights, &[f32], &[usize], &[f32], &LatentMoeConfig) -> Result<Vec<f32>>,
{
    let eps = ctx.eps;
    let mix = ablation.attnres_mix;
    let h = stream.mix(&layer.attn_res_proj, &layer.attn_res_norm, eps, mix);
    // Archive the *incoming* prefix (HF), not the post-mixer partial.
    stream.archive_incoming_at_block_start(layer.spec.index);
    let x = rms_norm(&h, &layer.input_norm, eps);
    let mix_out = match (&layer.mixer, mixer_state) {
        (MixerW::Kda(w), LayerCache::Kda(state)) => {
            kda_mixer(w, &x, ctx.kda, state, eps, ablation, &mut kda_decode)?
        }
        (MixerW::Mla(w), LayerCache::Mla(kv)) => mla_mixer(
            w,
            &x,
            ctx.mla,
            kv,
            pos,
            ctx.rope_theta,
            eps,
            ablation,
            &mut mla_decode,
        )?,
        _ => panic!(
            "K3 mixer/state mismatch at layer {} (CPU fallback GPU wrapper)",
            layer.spec.index
        ),
    };
    stream.add(&mix_out);
    let h = stream.mix(&layer.mlp_res_proj, &layer.mlp_res_norm, eps, mix);
    let x = rms_norm(&h, &layer.post_norm, eps);
    let mlp_out = match &layer.mlp {
        MlpW::Dense(w) => dense_mlp(
            w,
            &x,
            ctx.hidden,
            ctx.dense_intermediate,
            ctx.situ_beta,
            ctx.situ_linear_beta,
        ),
        MlpW::Moe(w) => moe_mlp_with(w, &x, ctx, ablation.force_expert, &mut moe_experts)?,
    };
    stream.add(&mlp_out);
    Ok(())
}

fn apply_o_proj(w: &[f32], x: &[f32], out: usize, inn: usize, ablation: Ablation) -> Vec<f32> {
    matvec_column_tp(
        w,
        x,
        out,
        inn,
        ablation.o_proj_tp,
        ablation.drop_o_proj_rank,
    )
}

fn kda_mixer<F>(
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
    let q = matvec(&w.q_proj, x, qdim, x.len());
    let k = matvec(&w.k_proj, x, qdim, x.len());
    let v = matvec(&w.v_proj, x, qdim, x.len());
    let mut qkv = q;
    qkv.extend_from_slice(&k);
    qkv.extend_from_slice(&v);
    // HF: g = f_b_proj(f_a_proj(x)) — two linears, no SiLU on the bottleneck.
    let fa = matvec(&w.f_a, x, cfg.head_dim, x.len());
    let z = matvec(&w.f_b, &fa, qdim, cfg.head_dim);
    let gate = bounded_gate(
        &z,
        &w.dt_bias,
        &w.a_log,
        cfg.heads,
        cfg.head_dim,
        cfg.gate_lower_bound,
    );
    let beta = matvec(&w.b_proj, x, cfg.heads, x.len());
    let g = matvec(&w.g_proj, x, qdim, x.len());
    let core = kda_decode(&qkv, &w.conv, &gate, &beta, cfg, state)?;
    let gated = gated_o_norm(&core, &g, &w.o_norm, cfg.head_dim, eps);
    Ok(apply_o_proj(&w.o_proj, &gated, x.len(), qdim, ablation))
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
fn mla_mixer<F>(
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
    let qa = matvec(&w.q_a, x, cfg.q_lora_rank, x.len());
    let qa = rms_norm(&qa, &w.q_a_ln, eps);
    let mut q = matvec(&w.q_b, &qa, cfg.heads * qk, cfg.q_lora_rank);
    let kv_in = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
    let kv_lat = matvec(&w.kv_a, x, kv_in, x.len());
    let (c, pe) = kv_lat.split_at(cfg.kv_lora_rank);
    let c = rms_norm(c, &w.kv_a_ln, eps);
    let kvb_out = cfg.heads * (cfg.qk_nope_head_dim + cfg.v_head_dim);
    let kvb = matvec(&w.kv_b, &c, kvb_out, cfg.kv_lora_rank);
    let (k, v) = pack_mla_kv(&kvb, pe, cfg);
    let mut k = k;
    let g = matvec(&w.g_proj, x, cfg.heads * cfg.v_head_dim, x.len());
    let attn = mla_decode(&mut q, &mut k, &v, &g, kv, cfg, pos, theta)?;
    Ok(apply_o_proj(
        &w.o_proj,
        &attn,
        x.len(),
        cfg.heads * cfg.v_head_dim,
        ablation,
    ))
}

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

fn dense_mlp(
    w: &DenseMlp,
    x: &[f32],
    hidden: usize,
    inter: usize,
    situ_beta: f32,
    situ_linear_beta: f32,
) -> Vec<f32> {
    let gate = matvec(&w.gate, x, inter, hidden);
    let up = matvec(&w.up, x, inter, hidden);
    let mid = situ_glu_vec(&gate, &up, situ_beta, situ_linear_beta);
    matvec(&w.down, &mid, hidden, inter)
}

fn moe_mlp_with<F>(
    w: &MoeWeights,
    x: &[f32],
    ctx: &K3LayerCtx<'_>,
    force: Option<usize>,
    experts_fn: &mut F,
) -> Result<Vec<f32>>
where
    F: FnMut(&MoeWeights, &[f32], &[usize], &[f32], &LatentMoeConfig) -> Result<Vec<f32>>,
{
    let mut logits = matvec(&w.router, x, ctx.moe.n_routed, ctx.moe.hidden);
    if let Some(e) = force {
        logits.fill(0.0);
        logits[e] = 8.0;
    }
    let shared = w.shared.as_ref().map(|s| {
        dense_mlp(
            s,
            x,
            ctx.moe.hidden,
            ctx.moe.expert_hidden,
            ctx.situ_beta,
            ctx.situ_linear_beta,
        )
    });
    let latent = matvec(&w.down, x, ctx.moe.latent, ctx.moe.hidden);
    let (ids, weights) = sigmoid_topk(&logits, &w.bias, ctx.moe.top_k);
    let mixed = experts_fn(w, &latent, &ids, &weights, ctx.moe)?;
    let mixed = if ctx.moe.use_norm {
        rms_norm(&mixed, &w.norm, ctx.eps)
    } else {
        mixed
    };
    let mut out = matvec(&w.up, &mixed, ctx.moe.hidden, ctx.moe.latent);
    if let Some(s) = shared {
        for (o, ss) in out.iter_mut().zip(&s) {
            *o += ss;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attnres_archives_incoming_at_block_start() {
        let mut s = AttnResStream::new(2, 4);
        s.partial = vec![1.0, 2.0];
        s.archive_incoming_at_block_start(0);
        assert_eq!(s.completed, vec![vec![1.0, 2.0]]);
        assert_eq!(s.partial, vec![0.0, 0.0]);
        s.add(&[0.5, 0.25]);
        assert_eq!(s.partial, vec![0.5, 0.25]);
        assert_eq!(
            s.completed[0],
            vec![1.0, 2.0],
            "archive is embed, not embed+mixer"
        );
        s.archive_incoming_at_block_start(1);
        assert_eq!(s.completed.len(), 1, "non-boundary layer must not archive");
        s.archive_incoming_at_block_start(4);
        assert_eq!(s.completed.len(), 2);
        assert_eq!(s.completed[1], vec![0.5, 0.25]);
        assert_eq!(s.partial, vec![0.0, 0.0]);
    }
}
