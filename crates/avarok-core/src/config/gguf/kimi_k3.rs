// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 (`general.architecture = kimi-k3`) GGUF → Atlas `kimi_k3` config.
//!
//! Do not alias onto qwen3_6_moe or any other loader. Builds an HF-shaped
//! `text_config` JSON and reuses [`crate::config::parsers::parse_kimi_k3`].

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use super::GgufMeta;
use crate::config::ModelConfig;
use crate::config::parsers::parse_kimi_k3;

/// Build a validated `kimi_k3` [`ModelConfig`] from Unsloth/llama.cpp K3 GGUF
/// metadata (`kimi-k3.*` keys).
pub(super) fn config_from_kimi_k3_gguf(meta: &dyn GgufMeta) -> Result<ModelConfig> {
    let arch = meta
        .get_str("general.architecture")
        .unwrap_or("kimi-k3");
    let k = |suffix: &str| format!("{arch}.{suffix}");
    let req_u64 = |suffix: &str| -> Result<u64> {
        meta.get_u64(&k(suffix))
            .with_context(|| format!("Kimi K3 GGUF missing {arch}.{suffix}"))
    };
    let req_f64 = |suffix: &str| -> Result<f64> {
        meta.get_f64(&k(suffix))
            .with_context(|| format!("Kimi K3 GGUF missing {arch}.{suffix}"))
    };

    let n_layers = req_u64("block_count")? as usize;
    let hidden = req_u64("embedding_length")? as usize;
    let n_heads = req_u64("attention.head_count")? as usize;
    let vocab = req_u64("vocab_size")? as usize;
    let ctx = req_u64("context_length")? as usize;
    let inter = req_u64("feed_forward_length")? as usize;
    let n_experts = req_u64("expert_count")? as usize;
    let top_k = req_u64("expert_used_count")? as usize;
    let n_shared = req_u64("expert_shared_count")? as usize;
    let moe_ff = req_u64("expert_feed_forward_length")? as usize;
    let latent = req_u64("expert_latent_length")? as usize;
    let dense_k = req_u64("leading_dense_block_count")? as usize;
    let q_lora = req_u64("attention.q_lora_rank")? as usize;
    let kv_lora = req_u64("attention.kv_lora_rank")? as usize;
    let kda_head_dim = req_u64("kda.head_dim")? as usize;
    let conv = req_u64("ssm.conv_kernel")? as usize;
    let rope_dim = req_u64("rope.dimension_count")? as usize;
    let attn_res = req_u64("attn_res.block_size")? as usize;
    let situ_beta = req_f64("activation.situ_beta")? as f32;
    let situ_lin = req_f64("activation.situ_linear_beta")? as f32;
    let rope_theta = meta.get_f64(&k("rope.freq_base")).unwrap_or(10_000.0) as f32;
    let rms = meta
        .get_f64(&k("attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5) as f32;
    let gate_lo = meta.get_f64(&k("kda.gate_lower_bound")).unwrap_or(-5.0) as f32;
    let v_mla = meta
        .get_u64(&k("attention.value_length_mla"))
        .unwrap_or(128) as usize;
    let k_mla = meta
        .get_u64(&k("attention.key_length_mla"))
        .unwrap_or(192) as usize;

    let kv_flags = meta
        .get_u64_arr(&k("attention.head_count_kv"))
        .with_context(|| format!("Kimi K3 GGUF missing array {arch}.attention.head_count_kv"))?;
    if kv_flags.len() != n_layers {
        bail!(
            "Kimi K3: attention.head_count_kv len {} != block_count {n_layers}",
            kv_flags.len()
        );
    }
    // GGUF: 0 = KDA, 1 = MLA. Official lists are 1-based.
    let mut kda_layers = Vec::new();
    let mut full_attn_layers = Vec::new();
    for (i, flag) in kv_flags.iter().enumerate() {
        let one_based = i + 1;
        if *flag == 0 {
            kda_layers.push(one_based);
        } else if *flag == 1 {
            full_attn_layers.push(one_based);
        } else {
            bail!("Kimi K3: attention.head_count_kv[{i}]={flag} (expected 0 or 1)");
        }
    }

    let text = json!({
        "model_type": "kimi_k3",
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_heads,
        "vocab_size": vocab,
        "max_position_embeddings": ctx,
        "intermediate_size": inter,
        "rms_norm_eps": rms,
        "rope_theta": rope_theta,
        "num_experts": n_experts,
        "num_experts_per_token": top_k,
        "num_shared_experts": n_shared,
        "moe_intermediate_size": moe_ff,
        "first_k_dense_replace": dense_k,
        "q_lora_rank": q_lora,
        "kv_lora_rank": kv_lora,
        "attn_res_block_size": attn_res,
        "activation_situ_beta": situ_beta,
        "activation_situ_linear_beta": situ_lin,
        "latent_moe_use_norm": true,
        "moe_renormalize": meta.get_u64(&k("expert_weights_norm")).unwrap_or(1) != 0,
        "routed_scaling_factor": meta.get_f64(&k("expert_weights_scale")).unwrap_or(1.0),
        "qk_nope_head_dim": k_mla.saturating_sub(rope_dim),
        "qk_rope_head_dim": rope_dim,
        "v_head_dim": v_mla,
        "hidden_act": "silu",
        "linear_attn_config": {
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn_layers,
            "head_dim": kda_head_dim,
            "num_heads": n_heads,
            "short_conv_kernel_size": conv,
            "gate_lower_bound": gate_lo,
            "use_full_rank_gate": true
        },
        // Latent MoE width: Atlas reads `expert_latent_length` via moe fields
        // already set above; keep explicit for parsers that look here.
        "latent_size": latent,
    });

    let wrapper = json!({
        "model_type": "kimi_k3",
        "text_config": text,
    });
    let mut config = parse_kimi_k3(&wrapper.to_string())
        .context("parse_kimi_k3 from synthesized GGUF text_config")?;
    // GGUF name map emits `model.*` / `lm_head.weight` (no language_model. prefix).
    config.weight_prefix = String::new();
    config.nested_config = false;
    // Surface latent width if the HF parser left it unset.
    if config.moe_latent_size == 0 {
        config.moe_latent_size = latent;
    }
    let _ = Value::Null;
    Ok(config)
}
