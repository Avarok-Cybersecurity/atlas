// SPDX-License-Identifier: AGPL-3.0-only

//! Model capabilities — config-derived feature flags.
//!
//! Instead of checking `config.model_type == "qwen3_5_moe"` throughout the codebase,
//! call sites use `config.capabilities().has_moe_layers` etc. Adding a new model
//! only requires implementing the capability derivation, not updating every call site.

/// SSM (State Space Model) architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsmArchitecture {
    /// No SSM layers — pure attention (Mistral, Llama, etc.)
    None,
    /// Gated Delta Networks (Qwen3.5 family)
    Gdn,
    /// Mamba-2 (Nemotron-H family)
    Mamba2,
}

/// Attention architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Standard multi-head or grouped-query attention (Qwen3.5, Llama, etc.)
    Standard,
    /// Multi-head Latent Attention — compressed KV via low-rank projection.
    /// (Mistral Small 4, DeepSeek-V2/V3)
    Mla,
}

/// Feature flags derived from model config at parse time.
///
/// These replace scattered `is_nemotron_h()`, `is_qwen35()` checks with
/// model-agnostic predicates. New models get capabilities automatically
/// from their config — no code changes needed.
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// Model has SSM/recurrent layers (GDN or Mamba-2).
    pub has_ssm_layers: bool,
    /// Model has full attention layers.
    pub has_attention_layers: bool,
    /// Model has MoE (Mixture of Experts) layers.
    pub has_moe_layers: bool,
    /// Model supports `<think>` reasoning tokens.
    /// Derived from tokenizer vocabulary, not model type name.
    pub supports_thinking: bool,
    /// Model has vision encoder (multimodal).
    pub supports_vision: bool,
    /// Model has MTP (Multi-Token Prediction) draft head.
    pub has_mtp: bool,
    /// SSM architecture family (determines state layout).
    pub ssm_architecture: SsmArchitecture,
    /// Attention architecture (standard GQA vs MLA compressed latent).
    pub attention_type: AttentionType,
    /// Model wraps language model config in a nested field (e.g., `text_config`).
    pub has_nested_config: bool,
}

impl ModelCapabilities {
    /// Derive capabilities from a ModelConfig.
    ///
    /// This is the SSOT for model feature detection. When adding a new model,
    /// ensure its config fields populate the right capabilities here — then
    /// all downstream code works automatically.
    pub fn from_config(config: &super::config::ModelConfig) -> Self {
        use super::config::LayerType;

        let has_ssm = config
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::LinearAttention));
        let has_mamba2 = config.mamba_num_heads > 0 && config.mamba_head_dim > 0;
        let has_attention = config
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::FullAttention));
        let has_moe = config.num_experts > 0;
        let has_vision = config.vision.is_some();
        let has_mtp = config.mtp_num_hidden_layers > 0;
        let has_nested = config.nested_config;

        let ssm_arch = if has_mamba2 {
            SsmArchitecture::Mamba2
        } else if has_ssm {
            SsmArchitecture::Gdn
        } else {
            SsmArchitecture::None
        };

        Self {
            has_ssm_layers: has_ssm || has_mamba2,
            has_attention_layers: has_attention,
            has_moe_layers: has_moe,
            // Models with SSM layers support <think> tokens. Long-term this should
            // be derived from tokenizer vocabulary, not architecture.
            supports_thinking: has_ssm || has_mamba2,
            supports_vision: has_vision,
            has_mtp,
            ssm_architecture: ssm_arch,
            has_nested_config: has_nested,
            attention_type: if config.kv_lora_rank > 0 {
                AttentionType::Mla
            } else {
                AttentionType::Standard
            },
        }
    }
}
