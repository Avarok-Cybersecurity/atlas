// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side 0.40B-shaped (or tiny) K3 weights for the C1 CPU graph.

use super::kda::KdaConfig;
use super::latent_moe::LatentMoeConfig;
use super::layer::{K3Graph, K3LayerSpec, MixerKind, MlpKind};
use super::mla::MlaConfig;
use super::ops::{fill, ident, ones};

/// Graph mutants. Mix=0 / force-expert-0 are C1; skip-layer is C2.
#[derive(Clone, Copy, Debug)]
pub struct Ablation {
    pub attnres_mix: f32,
    pub force_expert: Option<usize>,
    /// Skip this layer index in `forward_token` (C2 known-bad).
    pub skip_layer: Option<usize>,
}

impl Default for Ablation {
    fn default() -> Self {
        Self {
            attnres_mix: 1.0,
            force_expert: None,
            skip_layer: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KdaWeights {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub conv: Vec<f32>,
    pub f_a: Vec<f32>,
    pub f_b: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub a_log: Vec<f32>,
    pub b_proj: Vec<f32>,
    pub g_proj: Vec<f32>,
    pub o_norm: Vec<f32>,
    pub o_proj: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct MlaWeights {
    pub q_a: Vec<f32>,
    pub q_a_ln: Vec<f32>,
    pub q_b: Vec<f32>,
    pub kv_a: Vec<f32>,
    pub kv_a_ln: Vec<f32>,
    pub kv_b: Vec<f32>,
    pub g_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct DenseMlp {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct MoeWeights {
    pub down: Vec<f32>,
    pub up: Vec<f32>,
    pub norm: Vec<f32>,
    pub router: Vec<f32>,
    pub bias: Vec<f32>,
    pub experts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub shared: Option<DenseMlp>,
}

#[derive(Clone, Debug)]
pub enum MixerW {
    Kda(KdaWeights),
    Mla(MlaWeights),
}

#[derive(Clone, Debug)]
pub enum MlpW {
    Dense(DenseMlp),
    Moe(MoeWeights),
}

#[derive(Clone, Debug)]
pub struct K3CpuLayer {
    pub spec: K3LayerSpec,
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub attn_res_proj: Vec<f32>,
    pub attn_res_norm: Vec<f32>,
    pub mlp_res_proj: Vec<f32>,
    pub mlp_res_norm: Vec<f32>,
    pub mixer: MixerW,
    pub mlp: MlpW,
}

/// CPU twin (or tiny stand-in) ready for greedy decode.
#[derive(Clone, Debug)]
pub struct K3CpuModel {
    pub graph: K3Graph,
    pub kda: KdaConfig,
    pub mla: MlaConfig,
    pub moe: LatentMoeConfig,
    pub dense_intermediate: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub vocab: usize,
    pub embed: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub output_res_proj: Vec<f32>,
    pub output_res_norm: Vec<f32>,
    pub layers: Vec<K3CpuLayer>,
}

impl K3CpuModel {
    /// 8-layer twin pattern, tiny dims. Deterministic; no HF weights.
    pub fn synthetic_tiny() -> Self {
        let hidden = 4;
        let graph = twin_pattern_graph(hidden);
        let kda = KdaConfig {
            heads: 1,
            head_dim: 2,
            conv_kernel: 4,
            gate_lower_bound: -5.0,
            use_full_rank_gate: true,
        };
        let mla = MlaConfig {
            heads: 1,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 2,
            v_head_dim: 2,
            q_lora_rank: 4,
            kv_lora_rank: 2,
            mla_use_nope: true,
            mla_use_output_gate: true,
        };
        let moe = LatentMoeConfig {
            hidden,
            latent: 4,
            expert_hidden: 4,
            n_routed: 2,
            top_k: 1,
            n_shared: 1,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            use_norm: true,
            renormalize: true,
        };
        Self::build(graph, kda, mla, moe, 4, 8, 1e-5, 10_000.0)
    }

    /// Same 8-layer twin pattern, slightly wider. Still cheap on CPU.
    pub fn synthetic_small() -> Self {
        let hidden = 8;
        let graph = twin_pattern_graph(hidden);
        let kda = KdaConfig {
            heads: 2,
            head_dim: 4,
            conv_kernel: 4,
            gate_lower_bound: -5.0,
            use_full_rank_gate: true,
        };
        let mla = MlaConfig {
            heads: 2,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 2,
            v_head_dim: 4,
            q_lora_rank: 8,
            kv_lora_rank: 4,
            mla_use_nope: true,
            mla_use_output_gate: true,
        };
        let moe = LatentMoeConfig {
            hidden,
            latent: 8,
            expert_hidden: 8,
            n_routed: 2,
            top_k: 1,
            n_shared: 1,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            use_norm: true,
            renormalize: true,
        };
        Self::build(graph, kda, mla, moe, 8, 16, 1e-5, 10_000.0)
    }
}

fn twin_pattern_graph(hidden: usize) -> K3Graph {
    let mixers = [
        MixerKind::Kda,
        MixerKind::Kda,
        MixerKind::Kda,
        MixerKind::Mla,
        MixerKind::Kda,
        MixerKind::Kda,
        MixerKind::Kda,
        MixerKind::Mla,
    ];
    let layers = mixers
        .iter()
        .enumerate()
        .map(|(i, m)| K3LayerSpec {
            index: i,
            mixer: *m,
            mlp: if i == 0 {
                MlpKind::Dense
            } else {
                MlpKind::LatentMoe
            },
        })
        .collect();
    K3Graph {
        layers,
        hidden,
        attn_res_block_size: 4,
        situ_beta: 4.0,
        situ_linear_beta: 25.0,
        use_full_rank_gate: true,
        mla_use_nope: true,
        mla_use_output_gate: true,
    }
}

impl K3CpuModel {
    #[allow(clippy::too_many_arguments)]
    fn build(
        graph: K3Graph,
        kda: KdaConfig,
        mla: MlaConfig,
        moe: LatentMoeConfig,
        dense_intermediate: usize,
        vocab: usize,
        eps: f32,
        rope_theta: f32,
    ) -> Self {
        let h = graph.hidden;
        let layers = graph
            .layers
            .iter()
            .map(|spec| synth_layer(spec, &kda, &mla, &moe, dense_intermediate))
            .collect();
        Self {
            graph,
            kda,
            mla,
            moe,
            dense_intermediate,
            eps,
            rope_theta,
            vocab,
            embed: fill(vocab * h, 1, 0.05),
            lm_head: fill(vocab * h, 2, 0.05),
            final_norm: ones(h),
            output_res_proj: fill(h, 3, 0.05),
            output_res_norm: ones(h),
            layers,
        }
    }
}

fn synth_layer(
    spec: &K3LayerSpec,
    kda: &KdaConfig,
    mla: &MlaConfig,
    moe: &LatentMoeConfig,
    dense_int: usize,
) -> K3CpuLayer {
    let h = moe.hidden;
    let seed = 10 + spec.index as u32 * 17;
    let mixer = match spec.mixer {
        MixerKind::Kda => MixerW::Kda(synth_kda(kda, h, seed)),
        MixerKind::Mla => MixerW::Mla(synth_mla(mla, h, seed)),
    };
    let mlp = match spec.mlp {
        MlpKind::Dense => MlpW::Dense(synth_dense(h, dense_int, seed + 1)),
        MlpKind::LatentMoe => MlpW::Moe(synth_moe(moe, seed + 2)),
    };
    K3CpuLayer {
        spec: *spec,
        input_norm: ones(h),
        post_norm: ones(h),
        attn_res_proj: fill(h, seed + 3, 0.05),
        attn_res_norm: ones(h),
        mlp_res_proj: fill(h, seed + 4, 0.05),
        mlp_res_norm: ones(h),
        mixer,
        mlp,
    }
}

fn synth_kda(kda: &KdaConfig, hidden: usize, seed: u32) -> KdaWeights {
    let q = kda.qkv_dim();
    KdaWeights {
        q_proj: ident(q, hidden),
        k_proj: ident(q, hidden),
        v_proj: ident(q, hidden),
        conv: fill(kda.conv_elems(), seed, 0.05),
        f_a: fill(kda.head_dim * hidden, seed + 1, 0.05),
        f_b: fill(q * kda.head_dim, seed + 2, 0.05),
        dt_bias: vec![0.0; q],
        a_log: vec![-1.0; kda.heads],
        b_proj: fill(kda.heads * hidden, seed + 3, 0.05),
        g_proj: fill(q * hidden, seed + 4, 0.05),
        o_norm: ones(kda.head_dim),
        o_proj: ident(hidden, q),
    }
}

fn synth_mla(mla: &MlaConfig, hidden: usize, seed: u32) -> MlaWeights {
    let qk = mla.heads * mla.qk_head_dim();
    let dv = mla.heads * mla.v_head_dim;
    let kv_in = mla.kv_lora_rank + mla.qk_rope_head_dim;
    let kv_b_out = mla.heads * (mla.qk_nope_head_dim + mla.v_head_dim);
    MlaWeights {
        q_a: ident(mla.q_lora_rank, hidden),
        q_a_ln: ones(mla.q_lora_rank),
        q_b: fill(qk * mla.q_lora_rank, seed, 0.05),
        kv_a: fill(kv_in * hidden, seed + 1, 0.05),
        kv_a_ln: ones(mla.kv_lora_rank),
        kv_b: fill(kv_b_out * mla.kv_lora_rank, seed + 2, 0.05),
        g_proj: fill(dv * hidden, seed + 3, 0.05),
        o_proj: ident(hidden, dv),
    }
}

fn synth_dense(h: usize, inter: usize, seed: u32) -> DenseMlp {
    DenseMlp {
        gate: fill(inter * h, seed, 0.05),
        up: fill(inter * h, seed + 1, 0.05),
        down: ident(h, inter),
    }
}

fn synth_moe(moe: &LatentMoeConfig, seed: u32) -> MoeWeights {
    let mut experts = Vec::with_capacity(moe.n_routed);
    for e in 0..moe.n_routed {
        let mut w3 = ident(moe.expert_hidden, moe.latent);
        // Expert 0 stays identity-ish; others scale the up-branch.
        if e > 0 {
            w3[0] = 1.0 + e as f32;
        }
        experts.push((
            ident(moe.expert_hidden, moe.latent),
            ident(moe.latent, moe.expert_hidden),
            w3,
        ));
    }
    let shared = (moe.n_shared > 0).then(|| synth_dense(moe.hidden, moe.expert_hidden, seed + 9));
    MoeWeights {
        down: ident(moe.latent, moe.hidden),
        up: ident(moe.hidden, moe.latent),
        norm: ones(moe.latent),
        router: fill(moe.n_routed * moe.hidden, seed, 0.2),
        bias: vec![0.0; moe.n_routed],
        experts,
        shared,
    }
}
