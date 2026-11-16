// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the Qwen3.5 / 3.6 / 3.8 GDN-hybrid part of the GGUF config
//! builder (`general.architecture` `qwen35` and `qwen35moe`): which GGUF
//! architectures belong to the family, and the hybrid fields that the generic
//! decoder path in the parent module does not fill.

use anyhow::{Context, Result, bail};

use super::{GgufMeta, ModelConfig};
use crate::config::LayerType;

/// The Qwen3.5-family GDN-hybrid GGUF architectures (see [`apply_qwen35_hybrid`]).
pub(super) fn is_qwen35_family(arch: &str) -> bool {
    matches!(arch, "qwen35" | "qwen3_5" | "qwen35moe")
}

/// Fill the GDN-hybrid fields of a Qwen3.5-family config (`qwen35`,
/// `qwen35moe`) from the `{arch}.*` metadata, the same fields the safetensors
/// path reads from `text_config` (dispatch.rs, `qwen3_5` arm).
///
/// | `ModelConfig` field | GGUF key / rule |
/// |---|---|
/// | `full_attention_interval`, `layer_types` | `full_attention_interval`; layer `i` is full attention iff `(i + 1) % interval == 0`, else GDN (llama.cpp's `recurrent_layer_arr`) |
/// | `linear_num_key_heads` | `ssm.group_count` |
/// | `linear_num_value_heads` | `ssm.time_step_rank` |
/// | `linear_key_head_dim` | `ssm.state_size` |
/// | `linear_value_head_dim` | `ssm.inner_size / ssm.time_step_rank` |
/// | `linear_conv_kernel_dim` | `ssm.conv_kernel` |
/// | `partial_rotary_factor` | `rope.dimension_count / head_dim` |
/// | `mrope_section`, `mrope_interleaved` | `rope.dimension_sections` (llama.cpp stores four, the fourth is 0 here); interleaved because llama.cpp runs these archs with IMROPE |
/// | `mtp_num_hidden_layers` | `nextn_predict_layers` |
///
/// `layer_types` must be explicit: `ModelCapabilities` derives "has GDN layers"
/// from it, not from `full_attention_interval`. Every key above is required.
/// A missing or inconsistent value is an error, never a guess, because a
/// wrongly laid out hybrid stack still produces fluent-looking text.
pub(super) fn apply_qwen35_hybrid(
    config: &mut ModelConfig,
    meta: &dyn GgufMeta,
    arch: &str,
    nextn_layers: usize,
) -> Result<()> {
    let k = |suffix: &str| format!("{arch}.{suffix}");
    let req = |suffix: &str| -> Result<usize> {
        meta.get_u64(&k(suffix))
            .map(|v| v as usize)
            .with_context(|| {
                format!("GGUF: Qwen3.5-family arch '{arch}' missing required key '{arch}.{suffix}'")
            })
    };
    let nonzero = |suffix: &str, v: usize| -> Result<usize> {
        if v == 0 {
            bail!("GGUF metadata key '{arch}.{suffix}' must be greater than zero");
        }
        Ok(v)
    };

    // Layer layout.
    let interval = nonzero("full_attention_interval", req("full_attention_interval")?)?;
    config.full_attention_interval = interval;
    config.layer_types = (0..config.num_hidden_layers)
        .map(|i| {
            if (i + 1).is_multiple_of(interval) {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    if !config.layer_types.contains(&LayerType::FullAttention) {
        bail!(
            "GGUF: '{arch}.full_attention_interval' ({interval}) puts no full-attention layer \
             in {} trunk layers",
            config.num_hidden_layers
        );
    }

    // GDN geometry.
    let num_k_heads = nonzero("ssm.group_count", req("ssm.group_count")?)?;
    let num_v_heads = nonzero("ssm.time_step_rank", req("ssm.time_step_rank")?)?;
    let inner = nonzero("ssm.inner_size", req("ssm.inner_size")?)?;
    let key_head_dim = nonzero("ssm.state_size", req("ssm.state_size")?)?;
    let conv_kernel = nonzero("ssm.conv_kernel", req("ssm.conv_kernel")?)?;
    if !num_v_heads.is_multiple_of(num_k_heads) {
        bail!(
            "GGUF: '{arch}.ssm.time_step_rank' ({num_v_heads}) must be a multiple of \
             '{arch}.ssm.group_count' ({num_k_heads})"
        );
    }
    if !inner.is_multiple_of(num_v_heads) {
        bail!(
            "GGUF: '{arch}.ssm.inner_size' ({inner}) must be a multiple of \
             '{arch}.ssm.time_step_rank' ({num_v_heads})"
        );
    }
    config.linear_num_key_heads = num_k_heads;
    config.linear_num_value_heads = num_v_heads;
    config.linear_key_head_dim = key_head_dim;
    config.linear_value_head_dim = inner / num_v_heads;
    config.linear_conv_kernel_dim = conv_kernel;

    // Full-attention head geometry. Atlas carries one `head_dim` for K and V.
    if let Some(v) = meta.get_u64(&k("attention.value_length"))
        && v as usize != config.head_dim
    {
        bail!(
            "GGUF: '{arch}.attention.value_length' ({v}) differs from \
             '{arch}.attention.key_length' ({}); ModelConfig carries one head_dim",
            config.head_dim
        );
    }

    // Partial RoPE and MRoPE.
    let rotary = nonzero("rope.dimension_count", req("rope.dimension_count")?)?;
    if rotary > config.head_dim {
        bail!(
            "GGUF: '{arch}.rope.dimension_count' ({rotary}) exceeds head_dim ({})",
            config.head_dim
        );
    }
    config.partial_rotary_factor = rotary as f64 / config.head_dim as f64;
    if let Some(sections) = meta.get_u64_arr(&k("rope.dimension_sections")) {
        // llama.cpp stores [T, H, W, extra]; Atlas has [T, H, W]. The extra
        // section is 0 for every Qwen3.5-family file; refuse anything else.
        if sections.len() < 3 || sections.len() > 4 || sections.get(3).is_some_and(|v| *v != 0) {
            bail!(
                "GGUF: '{arch}.rope.dimension_sections' {sections:?} must be [T, H, W] or \
                 [T, H, W, 0]"
            );
        }
        let section = [
            sections[0] as usize,
            sections[1] as usize,
            sections[2] as usize,
        ];
        let total: usize = section.iter().sum();
        if total > 0 {
            if total * 2 != rotary {
                bail!(
                    "GGUF: '{arch}.rope.dimension_sections' {sections:?} sum to {total}, but \
                     twice the sum must equal '{arch}.rope.dimension_count' ({rotary})"
                );
            }
            config.mrope_section = section;
            config.mrope_interleaved = true;
        }
    }

    config.mtp_num_hidden_layers = nextn_layers;

    // The MoE variant takes the Qwen3.6 kernel-target name exactly when the
    // safetensors path does (dispatch.rs): MRoPE-interleaved MoE.
    if config.model_type == "qwen3_5_moe"
        && config.mrope_interleaved
        && config.mrope_section.iter().sum::<usize>() > 0
    {
        config.model_type = "qwen3_6_moe".to_string();
    }
    Ok(())
}
