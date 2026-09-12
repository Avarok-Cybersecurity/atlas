// SPDX-License-Identifier: AGPL-3.0-only

//! One-token K3 CPU forward: AttnRes + mixer + MLP, then logits.

use super::attnres::{attnres_mix, rms_norm};
use super::cache::{HybridCache, MlaKv};
use super::cpu_weights::{
    Ablation, DenseMlp, K3CpuLayer, K3CpuModel, KdaWeights, MixerW, MlaWeights, MlpW, MoeWeights,
};
use super::kda::{KdaConfig, KdaState, bounded_gate, kda_decode_token};
use super::latent_moe::latent_moe_forward;
use super::mla::{MlaConfig, apply_output_gate, maybe_rope, sdpa_one};
use super::ops::{embed_token, matvec};
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

    fn mix(&self, query: &[f32], norm_w: &[f32], eps: f32, mix: f32) -> Vec<f32> {
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
    for layer in &model.layers {
        if ablation.skip_layer == Some(layer.spec.index) {
            continue;
        }
        forward_layer(model, layer, pos, cache, &mut stream, ablation);
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

fn forward_layer(
    model: &K3CpuModel,
    layer: &K3CpuLayer,
    pos: usize,
    cache: &mut HybridCache,
    stream: &mut AttnResStream,
    ablation: Ablation,
) {
    let eps = model.eps;
    let mix = ablation.attnres_mix;
    let h = stream.mix(&layer.attn_res_proj, &layer.attn_res_norm, eps, mix);
    // Archive the *incoming* prefix (HF), not the post-mixer partial.
    stream.archive_incoming_at_block_start(layer.spec.index);
    let x = rms_norm(&h, &layer.input_norm, eps);
    let mix_out = match &layer.mixer {
        MixerW::Kda(w) => {
            let state = cache.kda_mut(layer.spec.index).expect("KDA cache slot");
            kda_mixer(w, &x, &model.kda, state, eps)
        }
        MixerW::Mla(w) => {
            let kv = cache.mla_mut(layer.spec.index).expect("MLA cache slot");
            mla_mixer(w, &x, &model.mla, kv, pos, model.rope_theta, eps)
        }
    };
    stream.add(&mix_out);
    let h = stream.mix(&layer.mlp_res_proj, &layer.mlp_res_norm, eps, mix);
    let x = rms_norm(&h, &layer.post_norm, eps);
    let mlp_out = match &layer.mlp {
        MlpW::Dense(w) => dense_mlp(w, &x, model.graph.hidden, model.dense_intermediate, model),
        MlpW::Moe(w) => moe_mlp(w, &x, model, ablation.force_expert),
    };
    stream.add(&mlp_out);
}

fn kda_mixer(
    w: &KdaWeights,
    x: &[f32],
    cfg: &KdaConfig,
    state: &mut KdaState,
    eps: f32,
) -> Vec<f32> {
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
    let core = kda_decode_token(&qkv, &w.conv, &gate, &beta, cfg, state);
    let gated = gated_o_norm(&core, &g, &w.o_norm, cfg.head_dim, eps);
    matvec(&w.o_proj, &gated, x.len(), qdim)
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

fn mla_mixer(
    w: &MlaWeights,
    x: &[f32],
    cfg: &MlaConfig,
    kv: &mut MlaKv,
    pos: usize,
    theta: f32,
    eps: f32,
) -> Vec<f32> {
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
    maybe_rope(
        &mut q,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        pos,
        theta,
        cfg.mla_use_nope,
    );
    maybe_rope(
        &mut k,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        pos,
        theta,
        cfg.mla_use_nope,
    );
    kv.append(&k, &v);
    let g = matvec(&w.g_proj, x, cfg.heads * cfg.v_head_dim, x.len());
    let attn = sdpa_one(&q, &kv.k, &kv.v, kv.seq_len, cfg.heads, qk, cfg.v_head_dim);
    let attn = apply_output_gate(&attn, &g, cfg.mla_use_output_gate);
    matvec(&w.o_proj, &attn, x.len(), cfg.heads * cfg.v_head_dim)
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

fn dense_mlp(w: &DenseMlp, x: &[f32], hidden: usize, inter: usize, model: &K3CpuModel) -> Vec<f32> {
    let gate = matvec(&w.gate, x, inter, hidden);
    let up = matvec(&w.up, x, inter, hidden);
    let mid = situ_glu_vec(
        &gate,
        &up,
        model.graph.situ_beta,
        model.graph.situ_linear_beta,
    );
    matvec(&w.down, &mid, hidden, inter)
}

fn moe_mlp(w: &MoeWeights, x: &[f32], model: &K3CpuModel, force: Option<usize>) -> Vec<f32> {
    let mut logits = matvec(&w.router, x, model.moe.n_routed, model.moe.hidden);
    if let Some(e) = force {
        logits.fill(0.0);
        logits[e] = 8.0;
    }
    let shared = w
        .shared
        .as_ref()
        .map(|s| dense_mlp(s, x, model.moe.hidden, model.moe.expert_hidden, model));
    let shared_ref = shared.as_deref();
    let (y, _) = latent_moe_forward(
        x, &w.down, &w.up, &w.norm, &logits, &w.bias, &w.experts, shared_ref, &model.moe, model.eps,
    );
    y
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
