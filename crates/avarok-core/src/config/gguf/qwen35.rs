// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.5/3.6 GDN-hybrid GGUF config fill (`qwen35` / `qwen35moe`).
//!
//! llama.cpp ships these as `general.architecture = qwen35moe` (MoE + GDN +
//! MRoPE). Atlas must not map that string onto `qwen3moe` / `qwen3_5_moe`:
//! those loaders have no Gated DeltaNet path. Qwen3.6-35B-A3B is `qwen3_6_moe`.

use anyhow::{Context, Result, bail};

use super::GgufMeta;
use crate::config::{LayerType, ModelConfig};

/// Fill SSM / hybrid-attn / MRoPE / shared-expert fields that the generic
/// decoder builder leaves at struct defaults. Missing keys are errors: a
/// GDN graph built without them is silently wrong.
pub(super) fn apply_qwen35_hybrid(
    config: &mut ModelConfig,
    meta: &dyn GgufMeta,
    arch: &str,
) -> Result<()> {
    let k = |suffix: &str| format!("{arch}.{suffix}");
    let req_u64 = |suffix: &str| -> Result<u64> {
        meta.get_u64(&k(suffix))
            .with_context(|| format!("GGUF metadata missing required key '{arch}.{suffix}'"))
    };

    let interval = req_u64("full_attention_interval")? as usize;
    if interval == 0 {
        bail!("GGUF metadata key '{arch}.full_attention_interval' must be greater than zero");
    }
    config.full_attention_interval = interval;

    let nextn = meta.get_u64(&k("nextn_predict_layers")).unwrap_or(0) as usize;
    // llama.cpp's `block_count` is n_layer_all (text + NextN). First ship
    // drops MTP: subtract when the remainder still matches the hybrid cadence.
    if nextn > 0 && config.num_hidden_layers > nextn {
        let text = config.num_hidden_layers - nextn;
        if text.is_multiple_of(interval) {
            config.num_hidden_layers = text;
        }
    }
    config.mtp_num_hidden_layers = 0;

    config.layer_types = (0..config.num_hidden_layers)
        .map(|i| {
            if (i + 1).is_multiple_of(interval) {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();

    config.linear_conv_kernel_dim = req_u64("ssm.conv_kernel")? as usize;
    config.linear_num_key_heads = req_u64("ssm.group_count")? as usize;
    config.linear_key_head_dim = req_u64("ssm.state_size")? as usize;
    config.linear_num_value_heads = req_u64("ssm.time_step_rank")? as usize;
    let inner = req_u64("ssm.inner_size")? as usize;
    if config.linear_num_value_heads == 0
        || !inner.is_multiple_of(config.linear_num_value_heads)
        || config.linear_num_key_heads == 0
        || !config
            .linear_num_value_heads
            .is_multiple_of(config.linear_num_key_heads)
    {
        bail!(
            "GGUF '{arch}.ssm.*' geometry is inconsistent: group_count={}, \
             time_step_rank={}, inner_size={inner}",
            config.linear_num_key_heads,
            config.linear_num_value_heads
        );
    }
    config.linear_value_head_dim = inner / config.linear_num_value_heads;
    if config.linear_conv_kernel_dim == 0 || config.linear_key_head_dim == 0 {
        bail!(
            "GGUF '{arch}.ssm.conv_kernel' and '{arch}.ssm.state_size' must be greater than zero"
        );
    }

    if arch == "qwen35moe" {
        let shexp = req_u64("expert_shared_feed_forward_length")? as usize;
        if shexp == 0 {
            bail!(
                "GGUF metadata key '{arch}.expert_shared_feed_forward_length' must be greater than zero"
            );
        }
        config.shared_expert_intermediate_size = shexp;
        config.norm_topk_prob = true;
    }

    if let Some(rot) = meta.get_u64(&k("rope.dimension_count")) {
        if rot == 0 || config.head_dim == 0 || !config.head_dim.is_multiple_of(rot as usize) {
            bail!(
                "GGUF '{arch}.rope.dimension_count' ({rot}) must be a non-zero divisor of head_dim ({})",
                config.head_dim
            );
        }
        config.partial_rotary_factor = (rot as f64) / (config.head_dim as f64);
        config.rotary_dim = rot as usize;
    }

    if let Some(sections) = meta.get_u64_arr(&k("rope.dimension_sections")) {
        if sections.len() < 3 {
            bail!(
                "GGUF '{arch}.rope.dimension_sections' must have at least 3 entries (got {})",
                sections.len()
            );
        }
        config.mrope_section = [
            sections[0] as usize,
            sections[1] as usize,
            sections[2] as usize,
        ];
        config.mrope_interleaved = config.mrope_section.iter().sum::<usize>() > 0;
    }

    Ok(())
}

/// Fail-closed: Kimi K3 GGUF must never alias onto another Atlas model_type.
pub(super) fn refuse_kimi_k3(arch: &str) -> Result<()> {
    let lower = arch.to_ascii_lowercase();
    let is_k3 = matches!(lower.as_str(), "kimi-k3" | "kimi_k3" | "kimik3")
        || (lower.contains("kimi") && lower.contains("k3"));
    if is_k3 {
        bail!(
            "GGUF general.architecture '{arch}' is Kimi K3. Atlas does not load K3 from GGUF \
             (no GGUF→kernel path). Use the official MXFP4/BF16 safetensors checkpoint. \
             This mapping is fail-closed and will not alias onto another model_type."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{GgufConfigInputs, config_from_gguf};
    use super::GgufMeta;
    use crate::config::LayerType;
    use std::collections::HashMap;

    #[derive(Default)]
    struct Meta {
        u: HashMap<String, u64>,
        f: HashMap<String, f64>,
        s: HashMap<String, String>,
        uarr: HashMap<String, Vec<u64>>,
    }
    impl Meta {
        fn u(mut self, k: &str, v: u64) -> Self {
            self.u.insert(k.into(), v);
            self
        }
        fn f(mut self, k: &str, v: f64) -> Self {
            self.f.insert(k.into(), v);
            self
        }
        fn s(mut self, k: &str, v: &str) -> Self {
            self.s.insert(k.into(), v.into());
            self
        }
        fn uarr(mut self, k: &str, v: Vec<u64>) -> Self {
            self.uarr.insert(k.into(), v);
            self
        }
    }
    impl GgufMeta for Meta {
        fn get_u64(&self, k: &str) -> Option<u64> {
            self.u.get(k).copied()
        }
        fn get_f64(&self, k: &str) -> Option<f64> {
            self.f.get(k).copied()
        }
        fn get_str(&self, k: &str) -> Option<&str> {
            self.s.get(k).map(String::as_str)
        }
        fn get_arr_len(&self, _k: &str) -> Option<usize> {
            None
        }
        fn get_u64_arr(&self, k: &str) -> Option<Vec<u64>> {
            self.uarr.get(k).cloned()
        }
    }

    fn qwen35moe_35b() -> Meta {
        Meta::default()
            .s("general.architecture", "qwen35moe")
            .u("qwen35moe.embedding_length", 2048)
            .u("qwen35moe.block_count", 41)
            .u("qwen35moe.attention.head_count", 16)
            .u("qwen35moe.attention.head_count_kv", 2)
            .u("qwen35moe.attention.key_length", 256)
            .u("qwen35moe.context_length", 262144)
            .u("qwen35moe.vocab_size", 248320)
            .u("qwen35moe.expert_count", 256)
            .u("qwen35moe.expert_used_count", 8)
            .u("qwen35moe.expert_feed_forward_length", 512)
            .u("qwen35moe.expert_shared_feed_forward_length", 512)
            .u("qwen35moe.ssm.conv_kernel", 4)
            .u("qwen35moe.ssm.state_size", 128)
            .u("qwen35moe.ssm.group_count", 16)
            .u("qwen35moe.ssm.time_step_rank", 32)
            .u("qwen35moe.ssm.inner_size", 4096)
            .u("qwen35moe.full_attention_interval", 4)
            .u("qwen35moe.rope.dimension_count", 64)
            .u("qwen35moe.nextn_predict_layers", 1)
            .f("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6)
            .f("qwen35moe.rope.freq_base", 10_000_000.0)
            .uarr("qwen35moe.rope.dimension_sections", vec![11, 11, 10, 0])
    }

    #[test]
    fn qwen35moe_maps_to_qwen3_6_moe_and_fills_hybrid() {
        let m = qwen35moe_35b();
        let c = config_from_gguf(&GgufConfigInputs {
            meta: &m,
            token_embd_vocab: None,
            has_output_weight: true,
        })
        .unwrap();
        assert_eq!(c.model_type, "qwen3_6_moe");
        assert!(c.attn_gated);
        assert_eq!(
            c.num_hidden_layers, 40,
            "NextN block stripped from the text stack"
        );
        assert_eq!(c.mtp_num_hidden_layers, 0);
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.moe_intermediate_size, 512);
        assert_eq!(c.shared_expert_intermediate_size, 512);
        assert_eq!(c.full_attention_interval, 4);
        assert_eq!(c.layer_types.len(), 40);
        assert_eq!(c.layer_types[0], LayerType::LinearAttention);
        assert_eq!(c.layer_types[3], LayerType::FullAttention);
        assert_eq!(c.linear_conv_kernel_dim, 4);
        assert_eq!(c.linear_num_key_heads, 16);
        assert_eq!(c.linear_num_value_heads, 32);
        assert_eq!(c.linear_key_head_dim, 128);
        assert_eq!(c.linear_value_head_dim, 128);
        assert!((c.partial_rotary_factor - 0.25).abs() < 1e-12);
        assert_eq!(c.mrope_section, [11, 11, 10]);
        assert!(c.mrope_interleaved);
        assert!(c.norm_topk_prob);
    }

    #[test]
    fn qwen35moe_missing_ssm_is_an_error() {
        let mut m = qwen35moe_35b();
        m.u.remove("qwen35moe.ssm.inner_size");
        let err = config_from_gguf(&GgufConfigInputs {
            meta: &m,
            token_embd_vocab: None,
            has_output_weight: true,
        })
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("qwen35moe.ssm.inner_size"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn qwen3moe_does_not_become_qwen3_6() {
        let m = Meta::default()
            .s("general.architecture", "qwen3moe")
            .u("qwen3moe.embedding_length", 2048)
            .u("qwen3moe.block_count", 48)
            .u("qwen3moe.feed_forward_length", 768)
            .u("qwen3moe.attention.head_count", 32)
            .u("qwen3moe.attention.head_count_kv", 4)
            .u("qwen3moe.attention.key_length", 128)
            .u("qwen3moe.context_length", 32768)
            .u("qwen3moe.vocab_size", 151936)
            .u("qwen3moe.expert_count", 128)
            .u("qwen3moe.expert_used_count", 8)
            .u("qwen3moe.expert_feed_forward_length", 768);
        let c = config_from_gguf(&GgufConfigInputs {
            meta: &m,
            token_embd_vocab: None,
            has_output_weight: true,
        })
        .unwrap();
        assert_eq!(c.model_type, "qwen3_5_moe");
        assert!(!c.attn_gated);
        assert_eq!(c.linear_num_key_heads, 0);
    }

    #[test]
    fn kimi_k3_gguf_is_refused_without_aliasing() {
        for arch in ["kimi-k3", "kimi_k3", "kimik3", "kimi-k3-instruct"] {
            let m = Meta::default().s("general.architecture", arch);
            let err = config_from_gguf(&GgufConfigInputs {
                meta: &m,
                token_embd_vocab: None,
                has_output_weight: true,
            })
            .unwrap_err()
            .to_string();
            assert!(err.contains("fail-closed"), "arch={arch} unexpected: {err}");
            assert!(
                !err.to_ascii_lowercase().contains("qwen"),
                "must not alias: {err}"
            );
        }
    }
}
