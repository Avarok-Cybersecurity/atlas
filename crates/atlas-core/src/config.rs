// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use serde::Deserialize;

/// Deserialize a u32 that may be JSON null (treat null as 0).
fn nullable_u32<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<u32, D::Error> {
    Option::<u32>::deserialize(d).map(|v| v.unwrap_or(0))
}

/// Layer type in a hybrid transformer model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    LinearAttention,
    /// Standalone MoE FFN layer (Nemotron-H: no mixer, just expert routing + FFN).
    Moe,
}

/// Model configuration parsed from HuggingFace config.json.
///
/// Single source of truth for model dimensions. All kernel launch
/// parameters and buffer sizes derive from this struct.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    // ── Core dimensions ──
    pub hidden_size: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default)]
    pub vocab_size: usize,

    // ── Full attention ──
    #[serde(default)]
    pub num_attention_heads: usize,
    /// GQA: number of K/V heads (≤ `num_attention_heads`). MQA when 1.
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    /// Fraction of `head_dim` that gets RoPE-rotated. 1.0 = full RoPE,
    /// 0.5 = half-rotated (Phi-style). Default 1.0.
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,

    // ── Linear attention (SSM / GDN) ──
    // "linear" = the recurrent state-space / gated-delta-net pathway used
    // by hybrid models (Qwen3.5/3.6, Nemotron-Nano, MiniMax). Per-token
    // updates run in O(1) state instead of O(seq) attention.
    #[serde(default)]
    pub linear_num_key_heads: usize,
    #[serde(default)]
    pub linear_key_head_dim: usize,
    #[serde(default)]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub linear_value_head_dim: usize,
    /// 1D causal-conv kernel size on the SSM input (typically 3 or 4).
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,

    // ── MoE ──
    #[serde(default)]
    pub num_experts: usize,
    /// Top-K experts activated per token (the "A" in 35B-A3B = 3B
    /// active params).
    #[serde(default = "default_one")]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    /// Renormalize routing probabilities so the K active experts sum
    /// to 1 after top-K selection. Qwen3.5+ sets true; older Qwen2 MoE
    /// variants set false.
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// MoE block stride: layer `i` uses MoE iff `i % decoder_sparse_step
    /// == 0`. 1 = every layer is MoE. Mistral / DeepSeek-style stagger
    /// uses 2.
    #[serde(default = "default_one")]
    pub decoder_sparse_step: usize,

    // ── Hybrid layer layout ──
    /// Per-layer kind (FullAttention | LinearAttention | …) parsed from
    /// HF config. When empty, falls back to `full_attention_interval`.
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
    /// Stride for full-attention layers in hybrid models when
    /// `layer_types` is empty: every Nth layer is FullAttention, the
    /// rest LinearAttention. 1 = every layer is full attention.
    #[serde(default = "default_one")]
    pub full_attention_interval: usize,
    /// Gemma-4 hybrid-attention sliding window size (0 = full attention).
    /// Sliding layers only attend to the last `sliding_window` KV positions;
    /// full layers (every 6th in Gemma-4) ignore this (effectively 0).
    /// Parsed from HF config.json `sliding_window` field. Uses `nullable_u32`
    /// because Nemotron-H (and some other models) set it to `null` in JSON.
    #[serde(default, deserialize_with = "nullable_u32")]
    pub sliding_window: u32,

    // ── Position embeddings ──
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    // ── Normalization ──
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,

    // ── Tokenizer ──
    /// BOS token ID (null → 0 for models without explicit BOS).
    #[serde(default, deserialize_with = "nullable_u32")]
    pub bos_token_id: u32,
    #[serde(default, deserialize_with = "nullable_u32")]
    pub eos_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// CLI override (`--lm-head-dtype`) for LM-head quantization, set at serve time
    /// (not from config.json). `Some(true)` = force BF16 lm_head; `Some(false)` = force
    /// the model's quantized lm_head; `None` = use the model-config-driven default.
    /// Consumed by `skip_lm_head_quantization()`. Replaces the ATLAS_LMHEAD_BF16 env var.
    #[serde(default)]
    pub lm_head_bf16_override: Option<bool>,
    /// When `skip_lm_head_quantization()` == false, quantize the LM head to FP8
    /// (E4M3, per-row scales, decoded via `w8a16_gemv`) instead of NVFP4.
    /// Set by `--lm-head-dtype fp8`. Additive: leaves the NVFP4/BF16 paths
    /// byte-identical when false.
    #[serde(default)]
    pub lm_head_fp8: bool,

    // ── Model type ──
    #[serde(default)]
    pub model_type: String,

    // ── MTP ──
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,

    // ── Nemotron-H / Mamba-2 ──
    #[serde(default)]
    pub hybrid_override_pattern: String,
    #[serde(default)]
    pub mamba_num_heads: usize,
    #[serde(default)]
    pub mamba_head_dim: usize,
    #[serde(default)]
    pub ssm_state_size: usize,
    #[serde(default)]
    pub n_groups: usize,
    #[serde(default)]
    pub expand: usize,
    /// Nemotron-H uses `n_routed_experts` (mapped to `num_experts` in parse_config).
    #[serde(default)]
    pub n_routed_experts: usize,
    /// Nemotron-H uses `norm_eps` (mapped to `rms_norm_eps` in parse_config).
    #[serde(default)]
    pub norm_eps: f64,
    /// Nemotron-H conv kernel size (mapped to `linear_conv_kernel_dim` in parse_config).
    #[serde(default)]
    pub conv_kernel: usize,
    /// Nemotron-H shared expert intermediate (mapped to shared_expert_intermediate_size).
    #[serde(default)]
    pub moe_shared_expert_intermediate_size: usize,
    /// Nemotron-H routed scaling factor for expert outputs.
    #[serde(default = "default_one_f64")]
    pub routed_scaling_factor: f64,
    /// LatentMoE: latent projection dimension for routed experts (Super 120B).
    /// When present, routed experts operate in latent space `[moe_latent_size]`
    /// instead of full `[hidden_size]`. Absent for Nano 30B.
    #[serde(default)]
    pub moe_latent_size: usize,

    // ── MLA (Multi-head Latent Attention) — Mistral Small 4 / DeepSeek-V2+ ──
    /// KV latent dimension for compressed cache. 0 = standard attention (no MLA).
    #[serde(default)]
    pub kv_lora_rank: usize,
    /// Per-layer KV cache dimensions (num_kv_heads, head_dim). Populated by
    /// loaders for heterogeneous-attention models (e.g. Gemma-4 with sliding
    /// and full attention having different head counts and dims). Empty for
    /// homogeneous models.
    #[serde(default, skip_deserializing, skip_serializing)]
    pub kv_layer_dims: Vec<(usize, usize)>,
    /// Query latent dimension for low-rank Q projection. 0 = standard Q.
    #[serde(default)]
    pub q_lora_rank: usize,
    /// Non-rotary portion of Q/K per head (NoPE component).
    #[serde(default)]
    pub qk_nope_head_dim: usize,
    /// Rotary portion of Q/K per head (RoPE component).
    #[serde(default)]
    pub qk_rope_head_dim: usize,
    /// Value dimension per head (may differ from head_dim in MLA).
    #[serde(default)]
    pub v_head_dim: usize,

    // ── YaRN RoPE scaling (Mistral Small 4) ──
    /// YaRN scaling factor (`yarn.factor`). 0.0 = YaRN disabled, use plain RoPE.
    #[serde(default)]
    pub yarn_factor: f32,
    /// YaRN low-rotation cutoff (`yarn.alpha` in Mistral params,
    /// `beta_slow` in HF transformers terminology).
    #[serde(default)]
    pub yarn_beta_slow: f32,
    /// YaRN high-rotation cutoff (`yarn.beta` in Mistral params,
    /// `beta_fast` in HF transformers terminology).
    #[serde(default)]
    pub yarn_beta_fast: f32,
    /// YaRN original context length used for the correction range
    /// (`yarn.original_max_position_embeddings`).
    #[serde(default)]
    pub yarn_original_max_position_embeddings: usize,
    /// llama_4_scaling Q temperature beta (`llama_4_scaling.beta`).
    /// Q is multiplied by `1 + beta * log(1 + floor(pos / original_max_pos))`
    /// after RoPE. 0.0 = disabled. Mistral Small 4 uses 0.1.
    #[serde(default)]
    pub llama_4_scaling_beta: f32,
    /// llama_4_scaling original context length for the Q temperature scale.
    #[serde(default)]
    pub llama_4_scaling_original_max_position_embeddings: usize,

    // ── Vision (Qwen3-VL only) ──
    /// Vision encoder configuration parsed from `vision_config` in config.json.
    /// None for text-only models.
    #[serde(skip)]
    pub vision: Option<VisionConfig>,

    /// Advertised quantization format + algorithm + per-module ignore list.
    /// Populated from `config.json::quantization_config` or a sibling
    /// `hf_quant_config.json` at `parse_config` time. `None` for
    /// un-quantized BF16/FP16 checkpoints. Consumed by the `QuantFormat`
    /// dispatcher (`crates/spark-model/src/quant_format/`) to pick the
    /// correct on-disk loader without guessing from tensor names.
    #[serde(skip)]
    pub quantization_config: Option<QuantizationConfig>,

    // ── Architecture flags (set by parse_config, not from JSON) ──
    /// Whether Q projection includes an output gate (Q+Gate interleaved, 2x q_dim).
    /// False for Qwen3-VL, Nemotron-H, Mistral (ungated Q).
    #[serde(skip)]
    pub attn_gated: bool,
    /// Whether config.json wraps the LLM config in a nested field (e.g., `text_config`).
    /// Determines weight prefix auto-detection behavior.
    #[serde(skip)]
    pub nested_config: bool,
    /// MRoPE (multi-modal rotary position embedding) section sizes in
    /// `[T, H, W]` order. `[0, 0, 0]` = scalar RoPE (default for Qwen3.5
    /// and earlier). Qwen3.6 uses `[11, 11, 10]`. Summed × 2 == rotary_dim.
    #[serde(skip)]
    pub mrope_section: [usize; 3],
    /// MRoPE channel layout: `true` = round-robin `[T H W T H W …]` (Qwen3.6),
    /// `false` = contiguous `[T…T | H…H | W…W]` (Qwen3-VL non-interleaved).
    /// Ignored when `mrope_section == [0, 0, 0]`.
    #[serde(skip)]
    pub mrope_interleaved: bool,

    // ── Weight key prefix (set by parser for conditional generation models) ──
    #[serde(skip)]
    pub weight_prefix: String,

    // ── Expert Parallelism (set at runtime, not from config.json) ──
    #[serde(skip)]
    pub ep_rank: usize,
    #[serde(skip)]
    pub ep_world_size: usize,

    // ── Tensor Parallelism (set at runtime, not from config.json) ──
    /// TP rank within the TP sub-communicator. 0 if `tp_world_size==1`.
    #[serde(skip)]
    pub tp_rank: usize,
    /// Number of TP ranks. 1 = no TP. Composes with EP statically:
    /// attention/MLP weights are TP-sharded; MoE expert weights are EP-sharded.
    #[serde(skip)]
    pub tp_world_size: usize,

    // ── FP8 KV cache calibration (set at runtime from CLI) ──
    /// Number of warmup tokens for online FP8 KV scale calibration.
    /// 0 = disabled (use static scales from checkpoint or uncalibrated 1.0).
    #[serde(skip)]
    pub fp8_kv_calibration_tokens: usize,

    // ── Gemma-4 specific ──
    /// Final logit softcapping: logits = cap * tanh(logits / cap).
    /// 0.0 = disabled (default for all models except Gemma-4 which uses 30.0).
    #[serde(skip)]
    pub final_logit_softcapping: f32,
    /// Embedding scale factor: embeddings *= scale after lookup.
    /// 0.0 = disabled (default). Gemma models use sqrt(hidden_size).
    #[serde(skip)]
    pub embed_scale: f32,

    // ── MiniMax M2 specific ──
    /// MoE routing activation. "" = default softmax. "sigmoid" = DeepSeek-V3
    /// / MiniMax-M2 style: raw gate logits pass through sigmoid to produce
    /// per-expert scores in (0,1), independent (not normalized across
    /// experts). Top-k selection may use a bias term (see `moe_routing_bias`).
    #[serde(default)]
    pub scoring_func: String,
    /// If true, a per-expert `e_score_correction_bias` tensor is added to
    /// routing scores *for top-k selection only* (not dispatch weighting).
    /// This is the DeepSeek-V3 loss-free balancing trick. The bias tensor
    /// itself lives in the checkpoint (typically one `[num_experts]` vector
    /// per MoE layer).
    #[serde(default)]
    pub use_routing_bias: bool,
    /// QK normalization granularity. "" = none (Qwen3-Next default).
    /// "per_layer" = each attention layer has its own learned q_layernorm /
    /// k_layernorm weight of shape `[head_dim]`, applied after Q/K projection
    /// and before RoPE (MiniMax M2).
    #[serde(default)]
    pub qk_norm_type: String,
    /// Number of sequential MTP draft modules. 0 = no MTP. 1 = existing
    /// Atlas MTP path (Qwen3.5). 3 = MiniMax M2 (each module is a single
    /// transformer layer that predicts one future token).
    #[serde(default)]
    pub num_mtp_modules: usize,
    /// Transformer layers per MTP module. 1 for MiniMax M2 (3 modules × 1
    /// layer = 3 future-token predictors).
    #[serde(default)]
    pub mtp_transformer_layers: usize,
    /// Explicit rotary dimension from config (bypasses partial_rotary_factor
    /// computation). MiniMax M2 ships `rotary_dim: 64` while head_dim=128,
    /// so the rotary factor is 0.5 — we honor the explicit int value when
    /// present for byte-exact rope dim.
    #[serde(default)]
    pub rotary_dim: usize,
    /// Target-model layer indices to capture intermediate hidden states from
    /// for DFlash speculative decoding. Sourced from the drafter's
    /// `dflash_config.target_layer_ids` (e.g., `[1, 10, 19, 28, 37]` for
    /// Qwen3.6-35B-A3B-DFlash). Empty when DFlash is disabled — its presence
    /// gates `TransformerModel::dflash_hidden_save` allocation and the
    /// per-layer capture hooks. Order matters: shallow-to-deep concatenation
    /// is what the drafter's `fc` projection expects.
    #[serde(default)]
    pub dflash_capture_layers: Vec<usize>,
}

/// Advertised weight-quantization layout, as declared in the HF
/// `config.json`'s `quantization_config` block (or a sibling
/// `hf_quant_config.json`). This is the authoritative signal for
/// format dispatch — the `QuantFormat` trait prefers this over
/// tensor-name sniffing, matching the dispatch model used by vLLM /
/// TensorRT-LLM / SGLang.
///
/// `quant_method` is the serialization scheme:
///   * `"compressed-tensors"` — Neural Magic / llm-compressor. Uses
///     `weight_packed` + `weight_global_scale` + `input_global_scale`.
///     Commonly paired with `format = "nvfp4-pack-quantized"` or
///     `"float-quantized"`.
///   * `"modelopt"` — NVIDIA TensorRT ModelOpt. Uses `weight` (as the
///     packed FP4 payload when `quant_algo == "NVFP4"`) + `weight_scale`
///     + `weight_scale_2` + `input_scale`.
///   * `"fp8"` — native FP8 block-scaled (e.g. `Qwen/Qwen3.5-35B-A3B-FP8`)
///     with `weight_scale_inv` sibling tensors.
///
/// `ignore_modules` holds the already-expanded list of module-path
/// patterns that should be loaded as dense BF16 rather than quantized.
/// Patterns use HF glob semantics (`*` matches any non-`.` sub-path).
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Raw `quant_method` string from the config. Stable values:
    /// `"compressed-tensors"`, `"modelopt"`, `"fp8"`.
    pub quant_method: String,
    /// ModelOpt-specific algorithm label: `"NVFP4"`, `"FP8"`, …
    /// Empty string for schemes that don't declare one (e.g. plain FP8).
    pub quant_algo: String,
    /// Optional `format` string (compressed-tensors uses this for
    /// `"nvfp4-pack-quantized"` and friends).
    pub format: String,
    /// Module-path globs that should stay BF16 (the "ignore list" in
    /// ModelOpt terminology; `targets`/`exclude_modules` in compressed-
    /// tensors). Example entries: `"lm_head"`,
    /// `"model.layers.*.self_attn*"`.
    pub ignore_modules: Vec<String>,
}

/// Vision encoder configuration for Qwen3-VL models.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Number of ViT transformer blocks (depth=27).
    pub depth: usize,
    /// ViT hidden dimension (1152).
    pub hidden_size: usize,
    /// Number of attention heads (16).
    pub num_heads: usize,
    /// Spatial patch size in pixels (16).
    pub patch_size: usize,
    /// Temporal patch size: still images are replicated this many times (2).
    pub temporal_patch_size: usize,
    /// 2×2 spatial merge: this many patch-lengths merged into one token (2).
    pub spatial_merge_size: usize,
    /// ViT MLP intermediate size (4304).
    pub intermediate_size: usize,
    /// Projection output dimension = LLM hidden_size (2048).
    pub out_hidden_size: usize,
    /// Layer indices after which deepstack mergers are applied ([8, 16, 24]).
    pub deepstack_visual_indexes: Vec<usize>,
    /// Placeholder token ID that marks where vision embeddings get spliced
    /// into the text embedding stream. Qwen3-VL uses 151655; Qwen3.6 uses
    /// 248056. When 0 the runtime falls back to the legacy Qwen3-VL value.
    pub image_pad_token_id: u32,
}

impl VisionConfig {
    /// Dimension of the merger input (spatial_merge_size² × hidden_size).
    pub fn merger_input_size(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size * self.hidden_size
    }
}

pub(crate) fn default_one() -> usize {
    1
}
pub(crate) fn default_one_f64() -> f64 {
    1.0
}
pub(crate) fn default_rope_theta() -> f64 {
    10000.0
}
pub(crate) fn default_rms_eps() -> f64 {
    1e-6
}
pub(crate) fn default_partial_rotary() -> f64 {
    1.0
}
pub(crate) fn default_conv_kernel() -> usize {
    4
}

mod dispatch;
mod factory;
mod methods;
mod parsers;
#[cfg(test)]
mod tests;

pub use dispatch::parse_config;
pub(crate) use parsers::{parse_gemma4_params, parse_minimax_m2, parse_vision_config};
pub use parsers::{parse_mistral_params, parse_quantization_config};

pub(crate) fn finalize_config(config: &mut ModelConfig, raw: &serde_json::Value) -> Result<()> {
    if config.quantization_config.is_none() {
        config.quantization_config = parse_quantization_config(raw);
    }
    validate_config(config)
}

/// Post-parse validation for ModelConfig.
/// Checks layer_types length matches num_hidden_layers and SSM field consistency.
pub(crate) fn validate_config(config: &ModelConfig) -> Result<()> {
    if !config.layer_types.is_empty() && config.layer_types.len() != config.num_hidden_layers {
        anyhow::bail!(
            "layer_types length ({}) doesn't match num_hidden_layers ({}) in config.json",
            config.layer_types.len(),
            config.num_hidden_layers,
        );
    }

    let has_ssm =
        config.layer_types.contains(&LayerType::LinearAttention) || config.linear_num_key_heads > 0;
    if has_ssm && config.linear_num_key_heads == 0 && config.mamba_num_heads == 0 {
        anyhow::bail!(
            "SSM model detected but linear_num_key_heads is 0 in config.json. \
             This field is required for SSM/GDN layer initialization."
        );
    }

    Ok(())
}
