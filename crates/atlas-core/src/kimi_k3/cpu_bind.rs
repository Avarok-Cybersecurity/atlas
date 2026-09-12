// SPDX-License-Identifier: AGPL-3.0-only

//! Name → [`K3CpuModel`] field bind for unpacked K3 safetensors.

use std::collections::HashMap;

use anyhow::{Context, Result};

use super::cpu_weights::{
    DenseMlp, K3CpuLayer, K3CpuModel, KdaWeights, MixerW, MlaWeights, MlpW, MoeWeights,
};
use super::kda::KdaConfig;
use super::latent_moe::LatentMoeConfig;
use super::layer::{K3Graph, K3LayerSpec, MixerKind, MlpKind};
use super::mla::MlaConfig;
use crate::config::ModelConfig;

pub(super) fn text_key(prefix: &str, rest: &str) -> String {
    let p = prefix.trim_end_matches('.');
    if p.is_empty() {
        rest.to_string()
    } else {
        format!("{p}.{rest}")
    }
}

pub(super) fn kda_from(c: &ModelConfig) -> KdaConfig {
    KdaConfig {
        heads: c.linear_num_key_heads,
        head_dim: c.linear_key_head_dim,
        conv_kernel: c.linear_conv_kernel_dim.max(1),
        gate_lower_bound: c.linear_gate_lower_bound,
        use_full_rank_gate: c.use_full_rank_gate,
    }
}

pub(super) fn mla_from(c: &ModelConfig) -> MlaConfig {
    MlaConfig {
        heads: c.num_attention_heads,
        qk_nope_head_dim: c.qk_nope_head_dim,
        qk_rope_head_dim: c.qk_rope_head_dim,
        v_head_dim: c.v_head_dim,
        q_lora_rank: c.q_lora_rank,
        kv_lora_rank: c.kv_lora_rank,
        mla_use_nope: c.mla_use_nope,
        mla_use_output_gate: c.mla_use_output_gate,
    }
}

pub(super) fn moe_from(c: &ModelConfig) -> LatentMoeConfig {
    LatentMoeConfig {
        hidden: c.hidden_size,
        latent: c.moe_latent_size,
        expert_hidden: c.moe_intermediate_size,
        n_routed: c.num_experts,
        top_k: c.num_experts_per_tok,
        n_shared: c.n_shared_experts,
        situ_beta: c.activation_situ_beta,
        situ_linear_beta: c.activation_situ_linear_beta,
        use_norm: c.latent_moe_use_norm,
        renormalize: c.norm_topk_prob,
    }
}

pub(super) fn inventory(
    c: &ModelConfig,
    graph: &K3Graph,
    kda: &KdaConfig,
    mla: &MlaConfig,
    moe: &LatentMoeConfig,
) -> Vec<(String, usize)> {
    let p = c.weight_prefix.as_str();
    let h = c.hidden_size;
    let mut v = vec![
        (text_key(p, "model.embed_tokens.weight"), c.vocab_size * h),
        (text_key(p, "model.norm.weight"), h),
        (text_key(p, "model.output_attn_res_proj.weight"), h),
        (text_key(p, "model.output_attn_res_norm.weight"), h),
    ];
    if !c.tie_word_embeddings {
        v.push((text_key(p, "lm_head.weight"), c.vocab_size * h));
    }
    for spec in &graph.layers {
        v.extend(layer_inventory(p, spec, c, kda, mla, moe));
    }
    v
}

fn layer_inventory(
    prefix: &str,
    spec: &K3LayerSpec,
    c: &ModelConfig,
    kda: &KdaConfig,
    mla: &MlaConfig,
    moe: &LatentMoeConfig,
) -> Vec<(String, usize)> {
    let lp = text_key(prefix, &format!("model.layers.{}", spec.index));
    let h = c.hidden_size;
    let mut k = vec![
        (format!("{lp}.input_layernorm.weight"), h),
        (format!("{lp}.post_attention_layernorm.weight"), h),
        (format!("{lp}.mlp_res_norm.weight"), h),
        (format!("{lp}.mlp_res_proj.weight"), h),
        (format!("{lp}.self_attention_res_norm.weight"), h),
        (format!("{lp}.self_attention_res_proj.weight"), h),
    ];
    match spec.mixer {
        MixerKind::Kda => {
            let q = kda.qkv_dim();
            k.extend([
                (format!("{lp}.self_attn.g_proj.weight"), q * h),
                (format!("{lp}.self_attn.o_proj.weight"), h * q),
                (format!("{lp}.self_attn.A_log"), kda.heads),
                (format!("{lp}.self_attn.b_proj.weight"), kda.heads * h),
                (format!("{lp}.self_attn.dt_bias"), q),
                (format!("{lp}.self_attn.f_a_proj.weight"), kda.head_dim * h),
                (format!("{lp}.self_attn.f_b_proj.weight"), q * kda.head_dim),
                (
                    format!("{lp}.self_attn.k_conv1d.weight"),
                    q * kda.conv_kernel,
                ),
                (format!("{lp}.self_attn.k_proj.weight"), q * h),
                (format!("{lp}.self_attn.o_norm.weight"), kda.head_dim),
                (
                    format!("{lp}.self_attn.q_conv1d.weight"),
                    q * kda.conv_kernel,
                ),
                (format!("{lp}.self_attn.q_proj.weight"), q * h),
                (
                    format!("{lp}.self_attn.v_conv1d.weight"),
                    q * kda.conv_kernel,
                ),
                (format!("{lp}.self_attn.v_proj.weight"), q * h),
            ]);
        }
        MixerKind::Mla => {
            let qk = mla.heads * mla.qk_head_dim();
            let dv = mla.heads * mla.v_head_dim;
            let kv_in = mla.kv_lora_rank + mla.qk_rope_head_dim;
            let kv_b = mla.heads * (mla.qk_nope_head_dim + mla.v_head_dim);
            k.extend([
                (format!("{lp}.self_attn.g_proj.weight"), dv * h),
                (format!("{lp}.self_attn.o_proj.weight"), h * dv),
                (
                    format!("{lp}.self_attn.kv_a_layernorm.weight"),
                    mla.kv_lora_rank,
                ),
                (
                    format!("{lp}.self_attn.kv_a_proj_with_mqa.weight"),
                    kv_in * h,
                ),
                (
                    format!("{lp}.self_attn.kv_b_proj.weight"),
                    kv_b * mla.kv_lora_rank,
                ),
                (
                    format!("{lp}.self_attn.q_a_layernorm.weight"),
                    mla.q_lora_rank,
                ),
                (
                    format!("{lp}.self_attn.q_a_proj.weight"),
                    mla.q_lora_rank * h,
                ),
                (
                    format!("{lp}.self_attn.q_b_proj.weight"),
                    qk * mla.q_lora_rank,
                ),
            ]);
        }
    }
    match spec.mlp {
        MlpKind::Dense => {
            let inter = c.intermediate_size;
            k.extend([
                (format!("{lp}.mlp.down_proj.weight"), h * inter),
                (format!("{lp}.mlp.gate_proj.weight"), inter * h),
                (format!("{lp}.mlp.up_proj.weight"), inter * h),
            ]);
        }
        MlpKind::LatentMoe => moe_inventory(&mut k, &lp, c, moe, h),
    }
    k
}

fn moe_inventory(
    k: &mut Vec<(String, usize)>,
    lp: &str,
    c: &ModelConfig,
    moe: &LatentMoeConfig,
    h: usize,
) {
    let eh = moe.expert_hidden;
    let lat = moe.latent;
    let shared = if c.shared_expert_intermediate_size > 0 {
        c.shared_expert_intermediate_size
    } else {
        eh
    };
    k.extend([
        (
            format!("{lp}.block_sparse_moe.gate.weight"),
            moe.n_routed * h,
        ),
        (
            format!("{lp}.block_sparse_moe.routed_expert_down_proj.weight"),
            lat * h,
        ),
        (
            format!("{lp}.block_sparse_moe.routed_expert_norm.weight"),
            lat,
        ),
        (
            format!("{lp}.block_sparse_moe.routed_expert_up_proj.weight"),
            h * lat,
        ),
    ]);
    if c.use_routing_bias {
        k.push((
            format!("{lp}.block_sparse_moe.gate.e_score_correction_bias"),
            moe.n_routed,
        ));
    }
    if moe.n_shared > 0 {
        k.extend([
            (
                format!("{lp}.block_sparse_moe.shared_experts.down_proj.weight"),
                h * shared,
            ),
            (
                format!("{lp}.block_sparse_moe.shared_experts.gate_proj.weight"),
                shared * h,
            ),
            (
                format!("{lp}.block_sparse_moe.shared_experts.up_proj.weight"),
                shared * h,
            ),
        ]);
    }
    for e in 0..moe.n_routed {
        k.extend([
            (
                format!("{lp}.block_sparse_moe.experts.{e}.w1.weight"),
                eh * lat,
            ),
            (
                format!("{lp}.block_sparse_moe.experts.{e}.w2.weight"),
                lat * eh,
            ),
            (
                format!("{lp}.block_sparse_moe.experts.{e}.w3.weight"),
                eh * lat,
            ),
        ]);
    }
}

fn pull(got: &mut HashMap<String, Vec<f32>>, prefix: &str, rest: &str) -> Result<Vec<f32>> {
    let k = text_key(prefix, rest);
    got.remove(&k)
        .with_context(|| format!("internal: missing {k}"))
}

pub(super) fn assemble(
    c: &ModelConfig,
    graph: K3Graph,
    kda: KdaConfig,
    mla: MlaConfig,
    moe: LatentMoeConfig,
    got: &mut HashMap<String, Vec<f32>>,
) -> Result<K3CpuModel> {
    let p = c.weight_prefix.as_str();
    let mut layers = Vec::with_capacity(graph.layers.len());
    for spec in &graph.layers {
        layers.push(assemble_layer(p, spec, c, &moe, got)?);
    }
    let lm_head = match pull(got, p, "lm_head.weight") {
        Ok(w) => w,
        Err(_) if c.tie_word_embeddings => pull(got, p, "model.embed_tokens.weight")?,
        Err(e) => return Err(e),
    };
    let embed = if c.tie_word_embeddings {
        lm_head.clone()
    } else {
        pull(got, p, "model.embed_tokens.weight")?
    };
    Ok(K3CpuModel {
        graph,
        kda,
        mla,
        moe,
        dense_intermediate: c.intermediate_size,
        eps: c.rms_norm_eps as f32,
        rope_theta: c.rope_theta as f32,
        vocab: c.vocab_size,
        embed,
        lm_head,
        final_norm: pull(got, p, "model.norm.weight")?,
        output_res_proj: pull(got, p, "model.output_attn_res_proj.weight")?,
        output_res_norm: pull(got, p, "model.output_attn_res_norm.weight")?,
        layers,
    })
}

fn assemble_layer(
    prefix: &str,
    spec: &K3LayerSpec,
    c: &ModelConfig,
    moe: &LatentMoeConfig,
    got: &mut HashMap<String, Vec<f32>>,
) -> Result<K3CpuLayer> {
    let lp = format!("model.layers.{}", spec.index);
    let mixer = match spec.mixer {
        MixerKind::Kda => assemble_kda(prefix, &lp, got)?,
        MixerKind::Mla => assemble_mla(prefix, &lp, got)?,
    };
    let mlp = match spec.mlp {
        MlpKind::Dense => MlpW::Dense(DenseMlp {
            gate: pull(got, prefix, &format!("{lp}.mlp.gate_proj.weight"))?,
            up: pull(got, prefix, &format!("{lp}.mlp.up_proj.weight"))?,
            down: pull(got, prefix, &format!("{lp}.mlp.down_proj.weight"))?,
        }),
        MlpKind::LatentMoe => assemble_moe(prefix, &lp, c, moe, got)?,
    };
    Ok(K3CpuLayer {
        spec: *spec,
        input_norm: pull(got, prefix, &format!("{lp}.input_layernorm.weight"))?,
        post_norm: pull(
            got,
            prefix,
            &format!("{lp}.post_attention_layernorm.weight"),
        )?,
        attn_res_proj: pull(got, prefix, &format!("{lp}.self_attention_res_proj.weight"))?,
        attn_res_norm: pull(got, prefix, &format!("{lp}.self_attention_res_norm.weight"))?,
        mlp_res_proj: pull(got, prefix, &format!("{lp}.mlp_res_proj.weight"))?,
        mlp_res_norm: pull(got, prefix, &format!("{lp}.mlp_res_norm.weight"))?,
        mixer,
        mlp,
    })
}

fn assemble_kda(prefix: &str, lp: &str, got: &mut HashMap<String, Vec<f32>>) -> Result<MixerW> {
    let q = pull(got, prefix, &format!("{lp}.self_attn.q_conv1d.weight"))?;
    let k = pull(got, prefix, &format!("{lp}.self_attn.k_conv1d.weight"))?;
    let v = pull(got, prefix, &format!("{lp}.self_attn.v_conv1d.weight"))?;
    let mut conv = q;
    conv.extend(k);
    conv.extend(v);
    Ok(MixerW::Kda(KdaWeights {
        q_proj: pull(got, prefix, &format!("{lp}.self_attn.q_proj.weight"))?,
        k_proj: pull(got, prefix, &format!("{lp}.self_attn.k_proj.weight"))?,
        v_proj: pull(got, prefix, &format!("{lp}.self_attn.v_proj.weight"))?,
        conv,
        f_a: pull(got, prefix, &format!("{lp}.self_attn.f_a_proj.weight"))?,
        f_b: pull(got, prefix, &format!("{lp}.self_attn.f_b_proj.weight"))?,
        dt_bias: pull(got, prefix, &format!("{lp}.self_attn.dt_bias"))?,
        a_log: pull(got, prefix, &format!("{lp}.self_attn.A_log"))?,
        b_proj: pull(got, prefix, &format!("{lp}.self_attn.b_proj.weight"))?,
        g_proj: pull(got, prefix, &format!("{lp}.self_attn.g_proj.weight"))?,
        o_norm: pull(got, prefix, &format!("{lp}.self_attn.o_norm.weight"))?,
        o_proj: pull(got, prefix, &format!("{lp}.self_attn.o_proj.weight"))?,
    }))
}

fn assemble_mla(prefix: &str, lp: &str, got: &mut HashMap<String, Vec<f32>>) -> Result<MixerW> {
    Ok(MixerW::Mla(MlaWeights {
        q_a: pull(got, prefix, &format!("{lp}.self_attn.q_a_proj.weight"))?,
        q_a_ln: pull(got, prefix, &format!("{lp}.self_attn.q_a_layernorm.weight"))?,
        q_b: pull(got, prefix, &format!("{lp}.self_attn.q_b_proj.weight"))?,
        kv_a: pull(
            got,
            prefix,
            &format!("{lp}.self_attn.kv_a_proj_with_mqa.weight"),
        )?,
        kv_a_ln: pull(
            got,
            prefix,
            &format!("{lp}.self_attn.kv_a_layernorm.weight"),
        )?,
        kv_b: pull(got, prefix, &format!("{lp}.self_attn.kv_b_proj.weight"))?,
        g_proj: pull(got, prefix, &format!("{lp}.self_attn.g_proj.weight"))?,
        o_proj: pull(got, prefix, &format!("{lp}.self_attn.o_proj.weight"))?,
    }))
}

fn assemble_moe(
    prefix: &str,
    lp: &str,
    c: &ModelConfig,
    moe: &LatentMoeConfig,
    got: &mut HashMap<String, Vec<f32>>,
) -> Result<MlpW> {
    let bias = if c.use_routing_bias {
        pull(
            got,
            prefix,
            &format!("{lp}.block_sparse_moe.gate.e_score_correction_bias"),
        )?
    } else {
        vec![0.0; moe.n_routed]
    };
    let shared = if moe.n_shared > 0 {
        Some(DenseMlp {
            gate: pull(
                got,
                prefix,
                &format!("{lp}.block_sparse_moe.shared_experts.gate_proj.weight"),
            )?,
            up: pull(
                got,
                prefix,
                &format!("{lp}.block_sparse_moe.shared_experts.up_proj.weight"),
            )?,
            down: pull(
                got,
                prefix,
                &format!("{lp}.block_sparse_moe.shared_experts.down_proj.weight"),
            )?,
        })
    } else {
        None
    };
    let mut experts = Vec::with_capacity(moe.n_routed);
    for e in 0..moe.n_routed {
        experts.push((
            pull(
                got,
                prefix,
                &format!("{lp}.block_sparse_moe.experts.{e}.w1.weight"),
            )?,
            pull(
                got,
                prefix,
                &format!("{lp}.block_sparse_moe.experts.{e}.w2.weight"),
            )?,
            pull(
                got,
                prefix,
                &format!("{lp}.block_sparse_moe.experts.{e}.w3.weight"),
            )?,
        ));
    }
    Ok(MlpW::Moe(MoeWeights {
        down: pull(
            got,
            prefix,
            &format!("{lp}.block_sparse_moe.routed_expert_down_proj.weight"),
        )?,
        up: pull(
            got,
            prefix,
            &format!("{lp}.block_sparse_moe.routed_expert_up_proj.weight"),
        )?,
        norm: pull(
            got,
            prefix,
            &format!("{lp}.block_sparse_moe.routed_expert_norm.weight"),
        )?,
        router: pull(got, prefix, &format!("{lp}.block_sparse_moe.gate.weight"))?,
        bias,
        experts,
        shared,
    }))
}
