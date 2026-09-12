// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 (`kimi_k3` / inner `kimi_linear`) config parser.
//!
//! Official geometry is taken from `docs/k3/fixtures/moonshotai-Kimi-K3-config.json`.
//! Unwrap `text_config` like GLM-5.3, then canonicalise to `kimi_k3`. Do **not**
//! alias onto `deepseek_v3` or `glm5_next`.
//!
//! `linear_attn_config.{kda_layers,full_attn_layers}` in the HF JSON are
//! **1-based** (last full-attn index is 93 = `num_hidden_layers`). Trust those
//! lists; do not reconstruct the 3:1 pattern from modular arithmetic.

use anyhow::{Context, Result, bail};

use super::super::{LayerType, ModelConfig, finalize_config, parse_quantization_config};

fn text_config(raw: &serde_json::Value) -> &serde_json::Value {
    raw.get("text_config").unwrap_or(raw)
}

pub fn parse_kimi_k3(json: &str) -> Result<ModelConfig> {
    let raw: serde_json::Value =
        serde_json::from_str(json).context("Invalid JSON in Kimi K3 config.json")?;
    let text = text_config(&raw).clone();

    let mut text_for_struct = text.clone();
    if let Some(obj) = text_for_struct.as_object_mut() {
        obj.remove("layer_types");
    }
    let text_json =
        serde_json::to_string(&text_for_struct).context("re-serialize kimi_k3 text_config")?;
    let mut config: ModelConfig =
        serde_json::from_str(&text_json).context("Failed to parse kimi_k3 text_config")?;

    // Canonical family name. Inner object says `kimi_linear`.
    config.model_type = "kimi_k3".to_string();
    config.nested_config = raw.get("text_config").is_some();
    config.weight_prefix = "language_model".to_string();

    overlay_moe(&mut config, &text)?;
    overlay_mla(&mut config, &text)?;
    overlay_linear_attn(&mut config, &text)?;
    overlay_k3_flags(&mut config, &text)?;

    config.layer_types = build_layer_types(&text, config.num_hidden_layers)?;
    config.mlp_only_layers = build_mlp_only_layers(&text, config.num_hidden_layers)?;

    if config.head_dim == 0 {
        config.head_dim = config.linear_key_head_dim;
    }
    if config.head_dim == 0 {
        bail!("kimi_k3: head_dim resolved to 0 (KDA head_dim missing)");
    }
    let qk = config.qk_nope_head_dim + config.qk_rope_head_dim;
    config.partial_rotary_factor = if qk > 0 {
        config.qk_rope_head_dim as f64 / qk as f64
    } else {
        0.0
    };

    config.quantization_config =
        parse_quantization_config(&text).or_else(|| parse_quantization_config(&raw));

    finalize_config(&mut config, &raw).context("kimi_k3: finalize_config")?;
    validate_kimi_k3(&config)?;
    Ok(config)
}

fn overlay_moe(config: &mut ModelConfig, text: &serde_json::Value) -> Result<()> {
    if config.num_experts == 0 && config.n_routed_experts > 0 {
        config.num_experts = config.n_routed_experts;
    }
    if let Some(v) = text.get("num_experts_per_token").and_then(|v| v.as_u64()) {
        config.num_experts_per_tok = v as usize;
    }
    let n_shared = text
        .get("num_shared_experts")
        .or_else(|| text.get("n_shared_experts"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    config.n_shared_experts = n_shared;
    if config.shared_expert_intermediate_size == 0 && n_shared > 0 {
        config.shared_expert_intermediate_size = n_shared * config.moe_intermediate_size;
    }
    if let Some(v) = text
        .get("routed_expert_hidden_size")
        .and_then(|v| v.as_u64())
    {
        config.moe_latent_size = v as usize;
    }
    if let Some(v) = text.get("moe_renormalize").and_then(|v| v.as_bool()) {
        config.norm_topk_prob = v;
    }
    if let Some(s) = text
        .get("moe_router_activation_func")
        .and_then(|v| v.as_str())
    {
        config.scoring_func = s.to_string();
    } else if let Some(s) = text.get("scoring_func").and_then(|v| v.as_str()) {
        config.scoring_func = s.to_string();
    }
    match text.get("topk_method").and_then(|v| v.as_str()) {
        Some("noaux_tc") => config.use_routing_bias = true,
        Some(other) => bail!("kimi_k3: unsupported topk_method {other:?} (expected noaux_tc)"),
        None => {}
    }
    Ok(())
}

fn overlay_mla(config: &mut ModelConfig, text: &serde_json::Value) -> Result<()> {
    let req_usize = |key: &str| -> Result<usize> {
        text.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .with_context(|| format!("kimi_k3 config.json missing `{key}`"))
    };
    config.q_lora_rank = req_usize("q_lora_rank")?;
    config.kv_lora_rank = req_usize("kv_lora_rank")?;
    config.qk_nope_head_dim = req_usize("qk_nope_head_dim")?;
    config.qk_rope_head_dim = req_usize("qk_rope_head_dim")?;
    config.v_head_dim = req_usize("v_head_dim")?;
    config.mla_use_nope = text
        .get("mla_use_nope")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    config.mla_use_output_gate = text
        .get("mla_use_output_gate")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    config.attn_gated = config.mla_use_output_gate;
    Ok(())
}

fn overlay_linear_attn(config: &mut ModelConfig, text: &serde_json::Value) -> Result<()> {
    let lac = text
        .get("linear_attn_config")
        .context("kimi_k3: missing linear_attn_config")?;
    let g = |k: &str| lac.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
    if let Some(v) = g("num_heads") {
        config.linear_num_key_heads = v;
        config.linear_num_value_heads = v;
    }
    if let Some(v) = g("head_dim") {
        config.linear_key_head_dim = v;
        config.linear_value_head_dim = v;
    }
    if let Some(v) = g("short_conv_kernel_size") {
        config.linear_conv_kernel_dim = v;
    }
    match lac.get("gate_lower_bound").and_then(|v| v.as_f64()) {
        Some(v) => config.linear_gate_lower_bound = v as f32,
        // 0.40B twin omits the key. Production JSON always has -5.0.
        None => config.linear_gate_lower_bound = -5.0,
    }
    config.use_full_rank_gate = lac
        .get("use_full_rank_gate")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Ok(())
}

fn overlay_k3_flags(config: &mut ModelConfig, text: &serde_json::Value) -> Result<()> {
    config.attn_res_block_size = text
        .get("attn_res_block_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .context("kimi_k3: missing attn_res_block_size")?;
    if let Some(s) = text.get("hidden_act").and_then(|v| v.as_str()) {
        config.hidden_act = s.to_string();
    }
    if let Some(v) = text.get("activation_situ_beta").and_then(|v| v.as_f64()) {
        config.activation_situ_beta = v as f32;
    }
    if let Some(v) = text
        .get("activation_situ_linear_beta")
        .and_then(|v| v.as_f64())
    {
        config.activation_situ_linear_beta = v as f32;
    }
    config.latent_moe_use_norm = text
        .get("latent_moe_use_norm")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Ok(())
}

/// Convert HF layer-index lists to 0-based Atlas indices.
///
/// Official K3 lists are 1-based (`full_attn_layers` ends at 93). Decide that
/// from the **union** of kda + full — KDA's max is 91, so a per-list check
/// would leave KDA 1-based and MLA converted, overlapping at layer 3.
/// GLM-style 0-based lists (max == n_layers-1, may contain 0) pass through.
fn to_zero_based(
    indices: &[usize],
    n_layers: usize,
    one_based: bool,
    what: &str,
) -> Result<Vec<usize>> {
    if indices.is_empty() {
        bail!("kimi_k3: {what} is empty");
    }
    if one_based && indices.contains(&0) {
        bail!("kimi_k3: {what} mixes 0-based and 1-based indices");
    }
    let out: Vec<usize> = indices
        .iter()
        .map(|&i| if one_based { i.saturating_sub(1) } else { i })
        .collect();
    for i in &out {
        if *i >= n_layers {
            bail!("kimi_k3: {what} index {i} out of range for {n_layers} layers");
        }
    }
    Ok(out)
}

fn idx_list(text: &serde_json::Value, key: &str) -> Option<Vec<usize>> {
    text.get("linear_attn_config")?
        .get(key)?
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as usize)
                .collect()
        })
}

fn build_layer_types(text: &serde_json::Value, n_layers: usize) -> Result<Vec<LayerType>> {
    let kda = idx_list(text, "kda_layers").context("kimi_k3: missing kda_layers")?;
    let full = idx_list(text, "full_attn_layers").context("kimi_k3: missing full_attn_layers")?;
    let max = kda.iter().chain(full.iter()).copied().max().unwrap_or(0);
    let one_based = max == n_layers;
    let kda = to_zero_based(&kda, n_layers, one_based, "kda_layers")?;
    let full = to_zero_based(&full, n_layers, one_based, "full_attn_layers")?;
    if kda.len() + full.len() != n_layers {
        bail!(
            "kimi_k3: kda_layers ({}) + full_attn_layers ({}) != num_hidden_layers ({})",
            kda.len(),
            full.len(),
            n_layers
        );
    }
    let mut types = vec![None; n_layers];
    for i in kda {
        if types[i].is_some() {
            bail!("kimi_k3: layer {i} listed as BOTH kda and full attention");
        }
        types[i] = Some(LayerType::LinearAttention);
    }
    for i in full {
        if types[i].is_some() {
            bail!("kimi_k3: layer {i} listed as BOTH kda and full attention");
        }
        types[i] = Some(LayerType::FullAttention);
    }
    types
        .into_iter()
        .enumerate()
        .map(|(i, t)| t.context(format!("kimi_k3: layer {i} missing from kda/full lists")))
        .collect()
}

fn build_mlp_only_layers(text: &serde_json::Value, n_layers: usize) -> Result<Vec<usize>> {
    let k = text
        .get("first_k_dense_replace")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .context("kimi_k3: missing first_k_dense_replace")?;
    if k > n_layers {
        bail!("kimi_k3: first_k_dense_replace {k} exceeds num_hidden_layers {n_layers}");
    }
    Ok((0..k).collect())
}

fn validate_kimi_k3(config: &ModelConfig) -> Result<()> {
    let linear = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    let full = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    if linear == 0 || full == 0 {
        bail!("kimi_k3: degenerate layer map — {linear} KDA / {full} MLA");
    }
    if config.layer_types.last() != Some(&LayerType::FullAttention) {
        bail!("kimi_k3: last layer must be gated MLA (full attention)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse_config;

    const OFFICIAL: &str =
        include_str!("../../../../../docs/k3/fixtures/moonshotai-Kimi-K3-config.json");

    #[test]
    fn parse_kimi_k3_official_config() {
        let c = parse_config(OFFICIAL).expect("official K3 config.json");
        assert_eq!(c.model_type, "kimi_k3");
        assert_eq!(c.num_hidden_layers, 93);
        assert_eq!(c.hidden_size, 7168);
        assert_eq!(c.num_attention_heads, 96);
        assert_eq!(c.mlp_only_layers, vec![0]);

        let kda = c
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        let mla = c
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count();
        assert_eq!(kda, 69);
        assert_eq!(mla, 24);
        assert_eq!(c.layer_types[0], LayerType::LinearAttention);
        assert_eq!(c.layer_types[3], LayerType::FullAttention);
        assert_eq!(c.layer_types[92], LayerType::FullAttention);

        assert_eq!(c.num_experts, 896);
        assert_eq!(c.num_experts_per_tok, 16);
        assert_eq!(c.n_shared_experts, 2);
        assert_eq!(c.moe_latent_size, 3584);
        assert_eq!(c.moe_intermediate_size, 3072);
        assert_eq!(c.shared_expert_intermediate_size, 2 * 3072);
        assert!(c.latent_moe_use_norm);
        assert_eq!(c.scoring_func, "sigmoid");
        assert!(c.use_routing_bias);
        assert!(c.norm_topk_prob);

        assert_eq!(c.hidden_act, "situ");
        assert_eq!(c.activation_situ_beta, 4.0);
        assert_eq!(c.activation_situ_linear_beta, 25.0);
        assert_eq!(c.attn_res_block_size, 12);

        assert!(c.mla_use_nope);
        assert!(c.mla_use_output_gate);
        assert!(c.use_full_rank_gate);
        assert_eq!(c.linear_gate_lower_bound, -5.0);
        assert_eq!(c.linear_key_head_dim, 128);
        assert_eq!(c.linear_value_head_dim, 128);
        assert_eq!(c.linear_num_key_heads, 96);
        assert_eq!(c.linear_conv_kernel_dim, 4);
        assert_eq!(c.q_lora_rank, 1536);
        assert_eq!(c.kv_lora_rank, 512);
        assert_eq!(c.qk_nope_head_dim, 128);
        assert_eq!(c.qk_rope_head_dim, 64);
        assert_eq!(c.v_head_dim, 128);
        assert_eq!(c.weight_prefix, "language_model");
        assert!(c.nested_config);
        let qc = c.quantization_config.as_ref().expect("quant config");
        assert_eq!(qc.format, "mxfp4-pack-quantized");
        assert_eq!(qc.quant_method, "compressed-tensors");
    }

    #[test]
    fn parse_kimi_k3_canonicalizes_inner_kimi_linear() {
        let raw: serde_json::Value = serde_json::from_str(OFFICIAL).unwrap();
        let inner = text_config(&raw);
        let c = parse_kimi_k3(&inner.to_string()).unwrap();
        assert_eq!(c.model_type, "kimi_k3");
        assert_eq!(c.num_hidden_layers, 93);
    }

    #[test]
    fn parse_kimi_k3_one_based_kda_mla_split() {
        let c = parse_kimi_k3(OFFICIAL).unwrap();
        // HF 1,2,3 → 0,1,2 KDA; HF 4 → 3 MLA; HF 93 → 92 MLA.
        for i in 0..3 {
            assert_eq!(c.layer_types[i], LayerType::LinearAttention, "layer {i}");
        }
        assert_eq!(c.layer_types[3], LayerType::FullAttention);
    }

    #[test]
    fn parse_kimi_k3_refuses_wrapper_without_text_config() {
        let mut raw: serde_json::Value = serde_json::from_str(OFFICIAL).unwrap();
        raw.as_object_mut().unwrap().remove("text_config");
        let err = parse_kimi_k3(&raw.to_string()).unwrap_err().to_string();
        assert!(
            err.contains("linear_attn_config") || err.contains("text_config"),
            "expected a geometry miss, got: {err}"
        );
    }

    #[test]
    fn parse_kimi_k3_0_40b_twin() {
        const TWIN: &str =
            include_str!("../../../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json");
        let c = parse_config(TWIN).expect("0.40B twin");
        assert_eq!(c.model_type, "kimi_k3");
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.num_hidden_layers, 8);
        assert_eq!(c.num_experts, 8);
        assert_eq!(c.num_experts_per_tok, 2);
        assert_eq!(c.attn_res_block_size, 4);
        assert_eq!(c.linear_gate_lower_bound, -5.0);
        assert_eq!(c.layer_types.last(), Some(&LayerType::FullAttention));
        let kda = c
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        assert_eq!(kda, 6);
    }
}
