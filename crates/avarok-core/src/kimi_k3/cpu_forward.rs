// SPDX-License-Identifier: AGPL-3.0-only

//! One-token K3 CPU forward: AttnRes + mixer + MLP, then logits.

use anyhow::Result;

use super::attnres::rms_norm;
use super::cache::{HybridCache, LayerCache, MlaKv};
use super::cpu_weights::{Ablation, DenseMlp, K3CpuLayer, K3CpuModel, MixerW, MlpW, MoeWeights};
use super::kda::{KdaConfig, KdaState, kda_decode_token};
use super::latent_moe::{LatentMoeConfig, mix_routed_experts};
use super::mla::{MlaConfig, mla_decode_token};
use super::ops::{embed_token, matvec};

mod dense;
mod mixer;
mod moe;
mod stream;
pub use stream::AttnResStream;

/// After row-parallel `o_proj` / dense MLP `down`. None at TP=1.
pub type HiddenReduce<'a> = &'a dyn Fn(&mut [f32]) -> Result<()>;

/// Optional resident dense/shared MLP, with local TP intermediate width.
pub type DenseMlpCore<'a> =
    &'a dyn Fn(&DenseMlp, &[f32], usize, usize, f32, f32) -> Result<Vec<f32>>;

/// Resident BF16 GEMV. `op` is a suffix key (q_proj, routed_down, ...).
pub type GpuGemv<'a> = &'a dyn Fn(&str, &[f32], usize, usize) -> Result<Vec<f32>>;

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
    /// After row-parallel `o_proj` (and dense MLP `down`). None at TP=1.
    pub reduce_hidden: Option<HiddenReduce<'a>>,
    pub dense_mlp: Option<DenseMlpCore<'a>>,
    pub gpu_gemv: Option<GpuGemv<'a>>,
    /// Shared-expert intermediate (full, before /tp). 6144 production.
    pub shared_intermediate: usize,
    /// Rank in TP. Slice hidden for routed down/up (7168/8, not n_experts).
    pub tp_rank: usize,
    pub tp_world: usize,
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
            reduce_hidden: None,
            dense_mlp: None,
            gpu_gemv: None,
            shared_intermediate: 0,
            tp_rank: 0,
            tp_world: 1,
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
    let mut mix_out = match (&layer.mixer, mixer_state) {
        (MixerW::Kda(w), LayerCache::Kda(state)) => {
            mixer::kda_mixer(ctx, w, &x, ctx.kda, state, eps, ablation, &mut kda_decode)?
        }
        (MixerW::Mla(w), LayerCache::Mla(kv)) => mixer::mla_mixer(
            ctx,
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
    if let Some(f) = ctx.reduce_hidden {
        f(&mut mix_out)?;
    }
    stream.add(&mix_out);
    let h = stream.mix(&layer.mlp_res_proj, &layer.mlp_res_norm, eps, mix);
    let x = rms_norm(&h, &layer.post_norm, eps);
    let mlp_out = match &layer.mlp {
        MlpW::Dense(w) => {
            let mut y = dense::run(ctx, w, &x, ctx.dense_intermediate)?;
            if let Some(f) = ctx.reduce_hidden {
                f(&mut y)?;
            }
            y
        }
        MlpW::Moe(w) => moe::moe_mlp_with(w, &x, ctx, ablation.force_expert, &mut moe_experts)?,
    };
    stream.add(&mlp_out);
    Ok(())
}
